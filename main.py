"""
Programa principal.

Fluxo geral:
 1. lê os argumentos (arquivo ROM, endereço-base e arquivo de CPI opcionais);
 2. lê o arquivo linha a linha e decodifica cada palavra de 32 bits;
 3. imprime a listagem das instruções (R4);
 4. imprime as estatísticas por formato e classe (R5);
 5. M2: árvore de dependências e pipeline com/sem reordenação, com o CPI.

argv[1] = arquivo com as instruções (obrigatório)
argv[2] = endereço da primeira instrução (opcional, padrão 0)
argv[3] = arquivo com o CPI de cada classe (opcional)
"""

import sys

import riscv


def imprimirListagem(lista):
    """Listagem (R4): uma linha de campos e uma de assembly por instrução."""
    for i in lista:
        # Instrução inválida: imprime o motivo e não entra nas estatísticas.
        # Se a linha nem era um número, não há palavra para mostrar ("----------").
        if not i.valida:
            palavra = "----------" if i.motivo.startswith("linha") else riscv.formatarBinario(i.palavra)
            print("{} | {} | INVÁLIDA: {}\n".format(i.pc, palavra, i.motivo))
            continue

        # Linha principal: endereço | palavra | formato | mnemônico | campos
        print("{} | {} | {} | {:<6} | {}".format(
            i.pc, riscv.formatarBinario(i.palavra), i.formato, i.mnemonico, i.campos()))

        # Segunda linha: assembly com nomes ABI e, se houver, a pseudo-instrução equivalente
        ps = i.pseudo()
        print("    asm: " + i.asm() + ("   [pseudo: " + ps + "]" if ps else ""))
        print()


def imprimirEstatisticas(lista):
    """Relatório estatístico (R5)."""
    porFormato, porClasse, validas, invalidas = riscv.estatisticas(lista)
    print("=== Relatório estatístico ===")
    print("Instruções válidas: {} | inválidas: {}".format(validas, invalidas))
    # Percentual de cada formato sobre as válidas (evita divisão por zero se não houver nenhuma)
    for formato, qtd in porFormato.items():
        pct = 100.0 * qtd / validas if validas else 0
        print("Tipo {}: {:3d}  ({:.1f}%)".format(formato, qtd, pct))

    # Quantidade por classe (o CPI do pipeline sai no fim do M2, depois dos hazards)
    print()
    for classe, qtd in porClasse.items():
        print("{:<6} qtd={:3d}".format(classe, qtd))


def imprimirArvore(lista, raiz):
    """M2: dependências de cada instrução e a divisão em graus."""
    print()
    print("=== M2: Árvore de dependências ===")

    for instrucao in [raiz] + lista:
        asm = "RAIZ" if instrucao is raiz else instrucao.asm()
        ps = instrucao.pseudo()
        print("Instrução: {} {} | pc: {} | rd: {} | filhos: {}".format(
            asm, "   [pseudo: " + ps + "]" if ps else "", instrucao.pc,
            riscv.reg(instrucao.rd) if instrucao.rd > 0 else "none", len(instrucao.filhos)))
        for f in instrucao.filhos:
            ps = f.pseudo()
            print("  Filho: {} {} | pc: {} | rs1: {} | rs2: {}".format(
                f.asm(), "   [pseudo: " + ps + "]" if ps else "", f.pc,
                riscv.reg(f.rs1) if f.rs1 > 0 else "none",
                riscv.reg(f.rs2) if f.rs2 > 0 else "none"))

    for k, grau in enumerate(riscv.listarArvore(raiz)):
        print("\nGrau {}:\n".format(k))
        for instrucao in grau:
            asm = "RAIZ" if instrucao is raiz else instrucao.asm()
            print(asm + " | " + instrucao.pseudo())


def imprimirPrograma(programa, base):
    """Lista o programa com os novos endereços, os nops inseridos e o CPI do pipeline."""
    for endereco, texto, original in riscv.linhasPrograma(programa, base):
        if original is None:
            print("{} | {}".format(endereco, texto))
        else:
            print("{} | {:<60} | original: {}".format(endereco, texto, original))
    instrucoes, nops, ciclos, cpi = riscv.resumoPrograma(programa)
    print("Instruções: {} | nops: {} | ciclos: {} | CPI: {:.3f}".format(instrucoes, nops, ciclos, cpi))


def main(args):
    # Sem o arquivo ROM não há o que decodificar: mostra o modo de uso e sai
    if len(args) < 2:
        print("Uso: python main.py <arquivo_rom> [endereco_base] [arquivo_cpi]")
        return

    # Endereço-base configurável (padrão 0). int(x, 0) aceita "0x...", "0b..." e decimal.
    # A máscara corta para 32 bits, como o (int) do Java.
    base = int(args[2], 0) & riscv.MASCARA_32 if len(args) >= 3 else 0

    # Tabela de CPI por classe (valores padrão; o arquivo opcional sobrescreve)
    cpi = {"ALU": 1.0, "LOAD": 2.0, "STORE": 2.0, "BRANCH": 2.0, "JUMP": 2.0}
    if len(args) >= 4:
        riscv.lerCpi(args[3], cpi)

    lista, qtdHex, qtdBin = riscv.lerPrograma(args[1], base)

    print("Formato de entrada detectado: " + riscv.formatoDetectado(qtdHex, qtdBin))
    print("Endereço base: 0x{:08X}\n".format(base))

    imprimirListagem(lista)
    imprimirEstatisticas(lista)

    raiz = riscv.criarArvoreDeDependencias(lista)
    imprimirArvore(lista, raiz)

    # Hazards: só as instruções válidas vão para o pipeline
    programaValido = [i for i in lista if i.valida]
    invalidas = len(lista) - len(programaValido)
    if invalidas > 0:
        print("\n({} instrução(ões) inválida(s) ficaram de fora do pipeline)".format(invalidas))

    print("\n=== M2: Pipeline sem reordenação (forwarding: {}) ===".format("sim" if riscv.FORWARDING else "não"))
    imprimirPrograma(riscv.inserirNops(programaValido), base)

    print("\n=== M2: Pipeline com reordenação ===")
    imprimirPrograma(riscv.reordenar(programaValido), base)


if __name__ == "__main__":
    main(sys.argv)
