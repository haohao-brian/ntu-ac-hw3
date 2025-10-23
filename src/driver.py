import sys
from bril import parse_bril
from ssa_construct import construct_ssa
from ssa_to_llvm import bril_to_llvm


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Bril programs to LLVM IR")
    parser.add_argument("--input", type=str, help="Input Bril JSON file", default=None)
    parser.add_argument("--output", type=str, help="Output LLVM IR file", default=None)
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            json_input = handle.read()
    else:
        json_input = sys.stdin.read()

    program = parse_bril(json_input)

    for function in program.functions:
        construct_ssa(function)

    llvm_ir = bril_to_llvm(program)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(llvm_ir)
    else:
        print(llvm_ir)


if __name__ == "__main__":
    main()
