from bril import Program

def bril_to_llvm(program: Program) -> str:
    """
    Translate a Bril program in SSA form to LLVM IR.

    Args:
        program (Program): The Bril program represented as a Program object.

    Returns:
        str: The generated LLVM IR code as a string.
    """
    llvm_ir_lines = []

    # Join all lines into a single LLVM IR string
    llvm_ir = '\n'.join(llvm_ir_lines)
    return llvm_ir