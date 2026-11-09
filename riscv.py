"""
Tudo que é RISC-V: decodificação (classe Instr), leitura da ROM,
estatísticas e tratamento de hazards do pipeline (M2).
Nada aqui imprime: quem formata e imprime o resultado é o main.py.
"""

import re

# Nomes ABI dos 32 registradores (posição i = registrador xi).
# (PATTERSON; WATERMAN, 2019, p. 21)
ABI = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
]

FORMATOS = "RISBUJ"
CLASSES = ["ALU", "LOAD", "STORE", "BRANCH", "JUMP"]

# Python não tem int de 32 bits: esta máscara faz o papel do "int" do Java
MASCARA_32 = (1 << 32) - 1

# Pipeline de 5 estágios (IF ID EX MEM WB). O banco de registradores escreve na
# 1ª metade do ciclo e lê na 2ª, então quem lê pode estar no ID enquanto o
# produtor está no WB.
#  - sem forwarding: o consumidor precisa estar 3 posições depois do produtor (2 nops)
#  - com forwarding: só um load seguido de uso do resultado precisa de 1 nop
ESTAGIOS = 5
FORWARDING = False


# Transforma int em registrador
def reg(n: int) -> str:
    return ABI[n]


def formatarBinario(n: int, tamanho: int = 32) -> str:
    """int em string binária com prefixo 0b e tamanho mínimo fixo (32 bits por padrão).
    Negativos saem em complemento de 2 com 32 bits, igual ao Integer.toBinaryString do Java."""
    return "0b" + format(n & MASCARA_32, "0{}b".format(tamanho))


def comSinal(valor: int, bits: int) -> int:
    """Estende o sinal: interpreta os 'bits' de baixo de valor como complemento de 2."""
    if valor & (1 << (bits - 1)):
        return valor - (1 << bits)
    return valor


# Estrutura que representa uma instrução decodificada
class Instr:

    def __init__(self, palavra: int | None, pc: int):
        """Decodifica a palavra de 32 bits. palavra=None cria uma instrução vazia (inválida)."""
        self.pc = pc                  # endereço da instrução
        self.palavra = palavra or 0   # palavra original de 32 bits
        self.valida = False           # False = não é uma instrução conhecida
        self.motivo = ""              # explicação, quando inválida
        self.formato = "?"            # R, I, S, B, U ou J
        self.mnemonico = None         # add, sub, lw, beq ...
        self.classe = None            # ALU, LOAD, STORE, BRANCH ou JUMP

        # Campos: valor -1 significa "este campo NÃO existe neste formato".
        self.rd = self.rs1 = self.rs2 = self.f3 = self.f7 = -1
        self.temImm = False
        self.imm = 0                  # imediato já reconstruído e com sinal estendido
        self.temAlvo = False
        self.alvo = 0                 # endereço absoluto de destino (só B e J)

        # Árvore de dependências e reordenação (M2)
        self.pais = [None, None]      # quem produziu rs1 e rs2 (RAW): decide os nops
        self.filhos = []              # instruções que precisam continuar DEPOIS desta
        self.antecessores = []        # instruções que precisam continuar ANTES desta
        self.grau = -1                # nível na árvore (-1 = ainda não calculado)
        self.ehAlvo = False           # algum desvio/salto pula para esta instrução
        self.inserida = False         # True = nop colocado pelo tratamento de hazards

        if palavra is None:
            return
        w = palavra

        # Campos que têm SEMPRE a mesma posição nos bits (mas nem sempre existem!)
        # (PATTERSON; WATERMAN, 2019, p. 17)
        opcode =  w        & 0b01111111   # bits 6..0
        rdv    = (w >> 7)  & 0b00011111   # bits 11..7
        f3     = (w >> 12) & 0b00000111   # bits 14..12
        rs1v   = (w >> 15) & 0b00011111   # bits 19..15
        rs2v   = (w >> 20) & 0b00011111   # bits 24..20
        f7     = (w >> 25) & 0b01111111   # bits 31..25

        # PASSO A: descobrir formato e mnemônico usando opcode + funct3 + funct7.
        # Se o mnemônico ficar None, a instrução é inválida.
        # (PATTERSON; WATERMAN, 2019, p. 18)
        match opcode:
            case 0b0110011:  # tipo R (add, sub, and, mul ...)
                self.formato = "R"
                if f7 == 0b0000000:
                    self.mnemonico = ["add", "sll", "slt", "sltu", "xor", "srl", "or", "and"][f3]
                elif f7 == 0b0100000:  # só sub e sra usam funct7 = 0100000
                    self.mnemonico = {0: "sub", 5: "sra"}.get(f3)
                elif f7 == 0b0000001:  # extensão M
                    self.mnemonico = ["mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"][f3]
                self.classe = "ALU"

            case 0b0010011:  # tipo I aritmético (addi, andi, slli ...)
                self.formato = "I"
                self.mnemonico = ["addi", None, "slti", "sltiu", "xori", None, "ori", "andi"][f3]
                if f3 == 1:
                    self.mnemonico = "slli" if f7 == 0b0000000 else None
                elif f3 == 5:  # srli e srai só se distinguem pelo funct7
                    self.mnemonico = {0b0000000: "srli", 0b0100000: "srai"}.get(f7)
                self.classe = "ALU"

            case 0b0000011:  # tipo I de carga (lb, lh, lw, lbu, lhu)
                self.formato = "I"
                self.mnemonico = ["lb", "lh", "lw", None, "lbu", "lhu", None, None][f3]
                self.classe = "LOAD"

            case 0b1100111:  # jalr
                self.formato = "I"
                self.mnemonico = "jalr" if f3 == 0 else None
                self.classe = "JUMP"

            case 0b1110011:  # ecall / ebreak (palavra inteira precisa bater)
                self.formato = "I"
                if w == 0b00000000000000000000000001110011:
                    self.mnemonico = "ecall"
                elif w == 0b00000000000100000000000001110011:
                    self.mnemonico = "ebreak"
                self.classe = "ALU"

            case 0b0100011:  # tipo S (sb, sh, sw)
                self.formato = "S"
                self.mnemonico = ["sb", "sh", "sw"][f3] if f3 < 3 else None
                self.classe = "STORE"

            case 0b1100011:  # tipo B (desvios condicionais)
                self.formato = "B"
                self.mnemonico = ["beq", "bne", None, None, "blt", "bge", "bltu", "bgeu"][f3]
                self.classe = "BRANCH"

            case 0b0110111:  # lui
                self.formato = "U"
                self.mnemonico = "lui"
                self.classe = "ALU"

            case 0b0010111:  # auipc
                self.formato = "U"
                self.mnemonico = "auipc"
                self.classe = "ALU"

            case 0b1101111:  # jal
                self.formato = "J"
                self.mnemonico = "jal"
                self.classe = "JUMP"

            case _:
                pass  # opcode desconhecido

        if self.mnemonico is None:
            self.formato = "?"
            self.classe = None
            self.motivo = "opcode/funct não reconhecido (opcode=0x{:02X}, f3={}, f7=0x{:02X})".format(
                opcode, f3, f7)
            return  # não preenche campo nenhum
        self.valida = True

        # PASSO B: preencher SOMENTE os campos que existem no formato.
        # (PATTERSON; WATERMAN, 2019, pp. 17-18)
        match self.formato:
            case "R":
                self.rd = rdv;  self.rs1 = rs1v;  self.rs2 = rs2v;  self.f3 = f3;  self.f7 = f7

            case "I":
                self.rd = rdv;  self.rs1 = rs1v;  self.f3 = f3
                self.temImm = True
                # bits 31..20, com sinal estendido
                self.imm = comSinal(w >> 20, 12)
                # Em slli/srli/srai o "imediato" é só o shamt (5 bits); o resto é funct7
                if self.mnemonico in ("slli", "srli", "srai"):
                    self.imm = (w >> 20) & 0b00011111

            case "S":
                self.rs1 = rs1v;  self.rs2 = rs2v;  self.f3 = f3
                self.temImm = True
                # imm[11:5] = bits 31..25  |  imm[4:0] = bits 11..7
                self.imm = comSinal(((w >> 25) << 5) | ((w >> 7) & 0b00011111), 12)

            case "B":
                # ATENÇÃO: os bits 11..7 NÃO são rd, fazem parte do imediato!
                self.rs1 = rs1v;  self.rs2 = rs2v;  self.f3 = f3
                self.temImm = True
                # imm[12]   = bit 31 (sinal)   imm[11]  = bit 7
                # imm[10:5] = bits 30..25      imm[4:1] = bits 11..8   imm[0] = 0
                self.imm = comSinal(((w >> 31) << 12)
                                    | (((w >> 7) & 0b1) << 11)
                                    | (((w >> 25) & 0b00111111) << 5)
                                    | (((w >> 8) & 0b00001111) << 1), 13)
                self.temAlvo = True
                self.alvo = pc + self.imm

            case "U":
                self.rd = rdv
                self.temImm = True
                # imm[31:12] = bits 31..12; os 12 bits baixos são zero
                self.imm = comSinal(w & 0b11111111111111111111000000000000, 32)

            case "J":
                self.rd = rdv
                self.temImm = True
                # imm[20]=bit 31 | imm[10:1]=bits 30..21 | imm[11]=bit 20 | imm[19:12]=bits 19..12
                self.imm = comSinal(((w >> 31) << 20)                  # sinal
                                    | (w & 0b11111111000000000000)     # imm[19:12] já está na posição certa
                                    | (((w >> 20) & 0b1) << 11)
                                    | (((w >> 21) & 0b1111111111) << 1), 21)
                self.temAlvo = True
                self.alvo = pc + self.imm

    @classmethod
    def linhaInvalida(cls, pc: int, linha: str) -> "Instr":
        """Linha do arquivo que nem é número: fica registrada como inválida."""
        i = cls(None, pc)
        i.motivo = 'linha não é hexadecimal nem binário: "{}"'.format(linha)
        return i

    @classmethod
    def criarNop(cls) -> "Instr":
        """nop = addi zero, zero, 0, marcado como inserido pelo tratamento de hazards."""
        n = cls(0b00000000000000000000000000010011, -1)
        n.inserida = True
        return n

    # 3. Desmontagem em assembly (nomes ABI)
    def asm(self) -> str:
        m = self.mnemonico
        match self.formato:
            case "R":
                return m + " " + reg(self.rd) + ", " + reg(self.rs1) + ", " + reg(self.rs2)
            case "I":
                if m in ("ecall", "ebreak"):
                    return m
                if self.classe == "LOAD" or m == "jalr":  # formato imm(rs1)
                    return "{} {}, {}({})".format(m, reg(self.rd), self.imm, reg(self.rs1))
                return "{} {}, {}, {}".format(m, reg(self.rd), reg(self.rs1), self.imm)
            case "S":
                return "{} {}, {}({})".format(m, reg(self.rs2), self.imm, reg(self.rs1))
            case "B":
                return m + " " + reg(self.rs1) + ", " + reg(self.rs2) + ", " + self.textoAlvo()
            case "U":
                return "{} {}, 0x{:x}".format(m, reg(self.rd), (self.imm >> 12) & 0b11111111111111111111)
            case "J":
                return m + " " + reg(self.rd) + ", " + self.textoAlvo()
            case _:
                return "?"

    def textoAlvo(self) -> str:
        """Texto do destino: endereço absoluto + deslocamento, ex.: "0b00001000 <PC-8>"."""
        desloc = "PC-{}".format(-self.imm) if self.imm < 0 else "PC+{}".format(self.imm)
        return "{} <{}>".format(formatarBinario(self.alvo, 8), desloc)

    # 4. Pseudo-instruções mais comuns (retorna "" se não houver)
    # (PATTERSON; WATERMAN, 2019, pp. 38-39)
    def pseudo(self) -> str:
        match self.mnemonico:
            case "addi":
                if self.rd == 0 and self.rs1 == 0 and self.imm == 0: return "nop"
                if self.rd != 0 and self.rs1 == 0: return "li {}, {}".format(reg(self.rd), self.imm)
                if self.rd != 0 and self.imm == 0: return "mv {}, {}".format(reg(self.rd), reg(self.rs1))
            case "xori":
                if self.imm == -1: return "not {}, {}".format(reg(self.rd), reg(self.rs1))
            case "sltiu":
                if self.imm == 1: return "seqz {}, {}".format(reg(self.rd), reg(self.rs1))
            case "sub":
                if self.rs1 == 0: return "neg {}, {}".format(reg(self.rd), reg(self.rs2))
            case "sltu":
                if self.rs1 == 0: return "snez {}, {}".format(reg(self.rd), reg(self.rs2))
            case "jalr":
                if self.rd == 0 and self.imm == 0:
                    return "ret" if self.rs1 == 1 else "jr " + reg(self.rs1)
            case "jal":
                if self.rd == 0: return "j " + self.textoAlvo()
            case "beq":
                if self.rs2 == 0: return "beqz {}, {}".format(reg(self.rs1), self.textoAlvo())
            case "bne":
                if self.rs2 == 0: return "bnez {}, {}".format(reg(self.rs1), self.textoAlvo())
        return ""

    def campos(self) -> str:
        """Lista de campos: só aparecem os que existem no formato."""
        partes = []
        if self.rd >= 0: partes.append("rd={}".format(self.rd))
        if self.rs1 >= 0: partes.append("rs1={}".format(self.rs1))
        if self.rs2 >= 0: partes.append("rs2={}".format(self.rs2))
        if self.temImm:
            # No tipo U o imediato é um valor de 32 bits (lui/auipc): fica mais legível em binário
            partes.append(formatarBinario(self.imm) if self.formato == "U" else "imm={}".format(self.imm))
        if self.f3 >= 0: partes.append("f3={}".format(self.f3))
        if self.f7 >= 0: partes.append("f7={}".format(self.f7))
        if self.temAlvo: partes.append(formatarBinario(self.alvo, 8))
        return " ".join(partes)

    # Instruções que escrevem em registrador (não zero)
    def escreve(self) -> bool:
        return self.valida and self.rd > 0

    # Instruções que leem de registrador r (não zero)
    def le(self, r: int) -> bool:
        return self.valida and r != 0 and (self.rs1 == r or self.rs2 == r)

    def ehMemoria(self) -> bool:
        return self.classe in ("LOAD", "STORE")

    def tamanhoAcesso(self) -> int:
        """Bytes lidos/escritos: f3 & 0b11 = 0 (byte), 1 (half), 2 (word)."""
        return [1, 2, 4][self.f3 & 0b11]

    def ehBarreira(self) -> bool:
        """Instruções que nada pode atravessar na reordenação."""
        if self.classe in ("BRANCH", "JUMP"): return True             # mudam o fluxo
        if self.mnemonico in ("ecall", "ebreak"): return True         # usam a0..a7 implicitamente
        return self.ehAlvo  # alguém pula para cá: o que está antes/depois tem que continuar antes/depois


# Leitura da entrada (R1)

def limparLinha(linha: str) -> str:
    """Remove comentários (#, // ou ;), espaços e o prefixo 0x."""
    corte = len(linha)
    for marca in ("#", "//", ";"):
        p = linha.find(marca)
        if 0 <= p < corte:
            corte = p
    s = linha[:corte].strip()
    if s.startswith(("0x", "0X")):
        s = s[2:]
    return s


def converter(s: str) -> int | None:
    """
    Converte o texto em palavra de 32 bits, detectando o formato sozinho:
     - exatamente 32 caracteres 0/1  -> binário
     - 1 a 8 dígitos hexadecimais    -> hexadecimal
    Retorna None se não for nenhum dos dois.
    """
    if re.fullmatch(r"[01]{32}", s): return int(s, 2)
    if re.fullmatch(r"[0-9a-fA-F]{1,8}", s): return int(s, 16)
    return None


def lerPrograma(caminho: str, base: int = 0) -> tuple[list[Instr], int, int]:
    """Lê o arquivo e decodifica cada linha. Devolve (instruções, qtdHex, qtdBin)."""
    lista = []           # todas as instruções, na ordem do arquivo
    pc = base            # endereço da instrução atual
    qtdHex = qtdBin = 0  # quantas linhas vieram em cada formato
    with open(caminho, encoding="utf-8", errors="replace") as arquivo:
        for bruta in arquivo:
            linha = limparLinha(bruta)
            if not linha:
                continue  # ignora linhas em branco e comentários

            w = converter(linha)  # None quando a linha não é hex nem binário
            if w is None:  # lixo no arquivo: registra como inválida e segue
                instrucao = Instr.linhaInvalida(pc, linha)
            else:
                # Mesmo critério de converter(): 32 caracteres 0/1 = binário, senão hex
                if len(linha) == 32 and re.fullmatch(r"[01]+", linha):
                    qtdBin += 1
                else:
                    qtdHex += 1
                instrucao = Instr(w, pc)
            lista.append(instrucao)
            pc += 4  # cada instrução ocupa 4 bytes (mesmo as inválidas, para manter os endereços)
    return lista, qtdHex, qtdBin


def lerCpi(arquivo: str, cpi: dict) -> None:
    """Lê o arquivo de CPI: linhas "CLASSE valor" (aceita vírgula decimal e comentários)."""
    with open(arquivo, encoding="utf-8", errors="replace") as f:
        for bruta in f:
            linha = limparLinha(bruta)
            if not linha:
                continue
            p = linha.split()
            if len(p) >= 2 and p[0].upper() in cpi:
                cpi[p[0].upper()] = float(p[1].replace(",", "."))


def formatoDetectado(qtdHex: int, qtdBin: int) -> str:
    """Formato de entrada encontrado (pode haver linhas dos dois tipos)."""
    if qtdBin > 0 and qtdHex > 0: return "MISTO"
    return "BINÁRIO" if qtdBin > 0 else "HEXADECIMAL"


# Relatório estatístico (R5)

def estatisticas(lista: list[Instr]) -> tuple[dict, dict, int, int]:
    """Devolve (porFormato, porClasse, validas, invalidas).
    Contadores começam em 0, assim os formatos/classes ausentes também saem no relatório."""
    porFormato = {f: 0 for f in FORMATOS}
    porClasse = {c: 0 for c in CLASSES}
    validas = invalidas = 0
    for i in lista:
        if not i.valida:
            invalidas += 1
            continue
        validas += 1
        porFormato[i.formato] += 1
        porClasse[i.classe] += 1
    return porFormato, porClasse, validas, invalidas


# M2: Árvore de dependências

def podeSerMesmoEndereco(a: Instr, b: Instr) -> bool:
    """Dois acessos à memória podem cair no mesmo byte?
    Bases diferentes: não dá para saber sem executar, então assume que sim.
    Mesma base: compara os intervalos [imm, imm + tamanho). Se a base mudou entre
    as duas, a dependência de registrador com quem a mudou já segura a ordem."""
    if a.rs1 != b.rs1:
        return True
    return a.imm < b.imm + b.tamanhoAcesso() and b.imm < a.imm + a.tamanhoAcesso()


def deveManterOrdem(a: Instr, b: Instr) -> bool:
    """a vem antes de b no programa original: b NÃO pode passar na frente de a?"""
    if a.ehBarreira() or b.ehBarreira(): return True
    if a.escreve() and b.le(a.rd): return True                     # RAW: b lê o que a escreveu
    if b.escreve() and a.le(b.rd): return True                     # WAR: b apagaria o valor que a ainda vai ler
    if a.escreve() and b.escreve() and a.rd == b.rd: return True   # WAW: o valor final tem que ser o de b
    # Memória: load/store não passa por store que pode usar o mesmo endereço
    if a.ehMemoria() and b.ehMemoria() and "STORE" in (a.classe, b.classe):
        return podeSerMesmoEndereco(a, b)
    return False


def criarArvoreDeDependencias(lista: list[Instr]) -> Instr:
    """Liga cada instrução às anteriores que ela não pode ultrapassar e devolve a raiz.
    Também guarda em pais[] quem produziu rs1/rs2 (RAW), que é o que decide os nops."""
    raiz = Instr(None, 0)

    # Marca as instruções que são destino de algum desvio/salto (início de bloco)
    for d in lista:
        if not d.temAlvo:
            continue
        for inst in lista:
            if inst.valida and inst.pc == d.alvo:
                inst.ehAlvo = True

    ultimoAEscrever = [None] * 32  # última instrução que escreveu cada registrador

    for i, instrucao in enumerate(lista):
        if not instrucao.valida:
            continue

        # Restrições de ordem: todas as anteriores que esta não pode ultrapassar
        for ant in lista[:i]:
            if ant.valida and deveManterOrdem(ant, instrucao):
                instrucao.antecessores.append(ant)
                ant.filhos.append(instrucao)
        if not instrucao.antecessores:
            raiz.filhos.append(instrucao)  # não depende de ninguém

        # Quem escreveu por último cada registrador lido (x0 não conta: vale sempre 0)
        if instrucao.rs1 > 0: instrucao.pais[0] = ultimoAEscrever[instrucao.rs1]
        if instrucao.rs2 > 0: instrucao.pais[1] = ultimoAEscrever[instrucao.rs2]
        if instrucao.escreve():
            ultimoAEscrever[instrucao.rd] = instrucao

    return raiz


def listarArvore(raiz: Instr) -> list[list[Instr]]:
    """Separa a árvore em graus. O grau é o caminho MAIS LONGO desde a raiz: uma
    instrução com vários pais fica logo abaixo do mais profundo, e aparece uma vez só.
    Uma instrução só é processada depois de todos os pais (ordem topológica)."""
    # Conta quantos pais cada instrução tem
    faltam = {}
    vistos = {raiz}
    pilha = [raiz]
    while pilha:
        n = pilha.pop()
        for f in n.filhos:
            faltam[f] = faltam.get(f, 0) + 1
            if f not in vistos:
                vistos.add(f)
                pilha.append(f)

    graus = []
    raiz.grau = 0
    fila = [raiz]
    while fila:
        n = fila.pop(0)
        while len(graus) <= n.grau:
            graus.append([])
        graus[n.grau].append(n)
        for f in n.filhos:
            f.grau = max(f.grau, n.grau + 1)
            faltam[f] -= 1
            if faltam[f] == 0:  # todos os pais já têm grau: o dele está definido
                fila.append(f)

    # Dentro de cada grau, mantém a ordem do programa
    for g in graus:
        g.sort(key=lambda inst: inst.pc)
    return graus


# M2: Tratamento de hazards

def distanciaMinima(p: Instr) -> int:
    """Distância mínima (em instruções) entre o produtor p e quem lê o resultado dele."""
    if not FORWARDING:
        return 3
    return 2 if p.classe == "LOAD" else 1


def nopsNecessarios(x: Instr, programa: list[Instr]) -> int:
    """Quantos nops precisam vir antes de x se ele for a próxima instrução do programa."""
    nops = 0
    for p in x.pais:
        if p is None or p not in programa:  # esse operando não depende de ninguém
            continue
        distancia = len(programa) - programa.index(p)
        nops = max(nops, distanciaMinima(p) - distancia)
    return nops


def inserirNops(ordem: list[Instr]) -> list[Instr]:
    """Mantém a ordem original e só coloca os nops onde há hazard."""
    programa = []
    for x in ordem:
        programa.extend(Instr.criarNop() for _ in range(nopsNecessarios(x, programa)))
        programa.append(x)
    return programa


def reordenar(lista: list[Instr]) -> list[Instr]:
    """
    Escalonamento de lista: a cada passo escolhe, entre as instruções prontas
    (todos os antecessores já colocados), a que precisa de menos nops.
    Empate: a que vem primeiro no programa original.
    """
    pendentes = list(lista)
    programa = []
    colocadas = set()
    while pendentes:
        escolhida = None
        menorNops = None
        # A primeira pendente sempre está pronta (todos os antecessores dela vêm antes)
        for x in pendentes:
            if not all(a in colocadas for a in x.antecessores):
                continue  # ainda espera alguém
            n = nopsNecessarios(x, programa)
            if menorNops is None or n < menorNops:
                menorNops = n
                escolhida = x
            if n == 0:
                break  # não tem como melhorar
        programa.extend(Instr.criarNop() for _ in range(menorNops))
        programa.append(escolhida)
        colocadas.add(escolhida)
        pendentes.remove(escolhida)
    return programa


def resumoPrograma(programa: list[Instr]) -> tuple[int, int, int, float]:
    """
    Devolve (instruções, nops, ciclos, CPI) do pipeline: cada instrução ou nop sai
    em 1 ciclo, mais (ESTAGIOS - 1) ciclos para encher o pipeline.
    Os nops são bolhas: contam ciclo, mas não instrução.
    """
    nops = sum(1 for i in programa if i.inserida)
    instrucoes = len(programa) - nops
    ciclos = (ESTAGIOS - 1) + len(programa)
    cpi = ciclos / instrucoes if instrucoes else 0
    return instrucoes, nops, ciclos, cpi


def cpiPipeline(programa: list[Instr]) -> float:
    return resumoPrograma(programa)[3]


def linhasPrograma(programa: list[Instr], base: int) -> list[tuple[int, str, int | None]]:
    """Programa com os novos endereços: (endereço, texto, pc original ou None se for nop inserido)."""
    linhas = []
    for k, i in enumerate(programa):
        endereco = base + 4 * k
        if i.inserida:
            linhas.append((endereco, "nop   <- inserido", None))
            continue
        texto = i.asm()
        # Desvios e saltos: mostra onde o destino foi parar no programa novo
        if i.temAlvo:
            for t, d in enumerate(programa):
                if not d.inserida and d.pc == i.alvo:
                    texto += "  (destino agora em {})".format(base + 4 * t)
        linhas.append((endereco, texto, i.pc))
    return linhas
