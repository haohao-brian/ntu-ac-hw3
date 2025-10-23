from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from bril import Const, EffectOperation, Function, Instruction, Label, Program, ValueOperation

LLVM_INT_TYPE = "i32"
LLVM_BOOL_TYPE = "i1"


@dataclass
class FunctionInfo:
    name: str
    return_type: str
    arg_types: List[str]


def bril_to_llvm(program: Program) -> str:
    """Translate a Bril program to LLVM IR."""
    builder = LLVMModuleBuilder(program)
    return builder.build()


class LLVMModuleBuilder:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.lines: List[str] = []
        self.function_infos: Dict[str, FunctionInfo] = {}
        self.llvm_names: Dict[str, str] = {}
        self._collect_function_info()

    def build(self) -> str:
        self._emit_prelude()
        for function in self.program.functions:
            self._emit_function(function)
        if any(func.name == "main" for func in self.program.functions):
            self._emit_main_wrapper()
        return "\n".join(self.lines)

    def _collect_function_info(self) -> None:
        for function in self.program.functions:
            return_type = llvm_type(function.type)
            arg_types = [llvm_type(arg["type"]) for arg in function.args]
            llvm_name = f"@bril_{function.name}"
            self.function_infos[function.name] = FunctionInfo(
                name=llvm_name, return_type=return_type, arg_types=arg_types
            )
            self.llvm_names[function.name] = llvm_name

    def _emit_prelude(self) -> None:
        self.lines.append("; ModuleID = 'bril_program'")
        self.lines.append("declare i32 @printf(i8*, ...)")
        self.lines.append("declare i32 @atoi(i8*)")
        self.lines.append(
            '@.int_fmt = private unnamed_addr constant [3 x i8] c"%d\\00"'
        )
        self.lines.append(
            '@.space_str = private unnamed_addr constant [2 x i8] c" \\00"'
        )
        self.lines.append(
            '@.newline_str = private unnamed_addr constant [2 x i8] c"\\0A\\00"'
        )
        self.lines.append(
            '@.true_str = private unnamed_addr constant [5 x i8] c"true\\00"'
        )
        self.lines.append(
            '@.false_str = private unnamed_addr constant [6 x i8] c"false\\00"'
        )

    def _emit_function(self, function: Function) -> None:
        info = self.function_infos[function.name]
        param_entries = []
        for idx, arg in enumerate(function.args):
            param_entries.append(
                f"{info.arg_types[idx]} %{sanitize_name(arg['name'])}.arg"
            )
        params = ", ".join(param_entries)
        self.lines.append(
            f"define {info.return_type} {info.name}({params}) {{"
        )
        temp_state = TempState()
        self.lines.append("entry:")

        var_types = collect_var_types(function)
        alloca_map: Dict[str, str] = {}
        for var, bril_ty in var_types.items():
            llvm_ty = llvm_type(bril_ty)
            ptr_name = f"%{sanitize_name(var)}.addr"
            alloca_map[var] = ptr_name
            self.lines.append(f"  {ptr_name} = alloca {llvm_ty}")

        for arg in function.args:
            llvm_ty = llvm_type(arg["type"])
            dest_ptr = alloca_map[arg["name"]]
            param_reg = f"%{sanitize_name(arg['name'])}.arg"
            self.lines.append(
                f"  store {llvm_ty} {param_reg}, {llvm_ty}* {dest_ptr}"
            )

        initial_instrs: List[Instruction] = []
        blocks: List[Tuple[str, List[Instruction]]] = []
        current_label: Optional[str] = None
        current_instrs: List[Instruction] = []

        for instr in function.instrs:
            if isinstance(instr, Label):
                if current_label is not None:
                    blocks.append((current_label, current_instrs))
                    current_instrs = []
                current_label = instr.label
            else:
                if current_label is None:
                    initial_instrs.append(instr)
                else:
                    current_instrs.append(instr)
        if current_label is not None:
            blocks.append((current_label, current_instrs))

        terminated = False
        for instr in initial_instrs:
            terminated |= emit_instruction(
                instr,
                self.lines,
                alloca_map,
                var_types,
                temp_state,
                self.llvm_names,
            )
        next_label = blocks[0][0] if blocks else None
        if not terminated:
            if next_label is not None:
                self.lines.append(
                    f"  br label %{sanitize_label(next_label)}"
                )
            else:
                self._emit_default_return(info.return_type)
                terminated = True

        for idx, (label, instrs) in enumerate(blocks):
            self.lines.append(f"{sanitize_label(label)}:")
            terminated = False
            for instr in instrs:
                terminated |= emit_instruction(
                    instr,
                    self.lines,
                    alloca_map,
                    var_types,
                    temp_state,
                    self.llvm_names,
                )
            if not terminated:
                next_label = (
                    sanitize_label(blocks[idx + 1][0])
                    if idx + 1 < len(blocks)
                    else None
                )
                if next_label is not None:
                    self.lines.append(f"  br label %{next_label}")
                else:
                    self._emit_default_return(info.return_type)
        self.lines.append("}")

    def _emit_default_return(self, return_type: str) -> None:
        if return_type == "void":
            self.lines.append("  ret void")
        elif return_type == LLVM_BOOL_TYPE:
            self.lines.append("  ret i1 0")
        else:
            self.lines.append(f"  ret {return_type} 0")

    def _emit_main_wrapper(self) -> None:
        function = next(func for func in self.program.functions if func.name == "main")
        info = self.function_infos["main"]
        wrapper = LLVMWrapperEmitter(function, info, self.llvm_names["main"])
        self.lines.extend(wrapper.emit())


class LLVMWrapperEmitter:
    def __init__(
        self, function: Function, info: FunctionInfo, target_name: str
    ) -> None:
        self.function = function
        self.info = info
        self.target_name = target_name
        self.temp_state = TempState()

    def emit(self) -> List[str]:
        lines: List[str] = []
        lines.append("define i32 @main(i32 %argc, i8** %argv) {")
        lines.append("entry:")
        arg_values: List[Tuple[str, str]] = []
        for index, arg in enumerate(self.function.args, start=1):
            llvm_ty = llvm_type(arg["type"])
            gep_reg = self.temp_state.new_tmp()
            lines.append(
                f"  {gep_reg} = getelementptr inbounds i8*, i8** %argv, i64 {index}"
            )
            load_reg = self.temp_state.new_tmp()
            lines.append(f"  {load_reg} = load i8*, i8** {gep_reg}")
            atoi_reg = self.temp_state.new_tmp()
            lines.append(f"  {atoi_reg} = call i32 @atoi(i8* {load_reg})")
            if llvm_ty == LLVM_BOOL_TYPE:
                bool_reg = self.temp_state.new_tmp()
                lines.append(f"  {bool_reg} = icmp ne i32 {atoi_reg}, 0")
                arg_values.append((llvm_ty, bool_reg))
            else:
                arg_values.append((llvm_ty, atoi_reg))
        call_args = ", ".join(f"{ty} {reg}" for ty, reg in arg_values)
        call_suffix = f"({call_args})" if call_args else "()"
        if self.info.return_type == "void":
            lines.append(f"  call void {self.target_name}{call_suffix}")
            lines.append("  ret i32 0")
        elif self.info.return_type == LLVM_BOOL_TYPE:
            call_reg = self.temp_state.new_tmp()
            lines.append(
                f"  {call_reg} = call i1 {self.target_name}{call_suffix}"
            )
            zext_reg = self.temp_state.new_tmp()
            lines.append(f"  {zext_reg} = zext i1 {call_reg} to i32")
            lines.append(f"  ret i32 {zext_reg}")
        else:
            call_reg = self.temp_state.new_tmp()
            lines.append(
                f"  {call_reg} = call {self.info.return_type} {self.target_name}{call_suffix}"
            )
            if self.info.return_type == LLVM_INT_TYPE:
                lines.append(f"  ret i32 {call_reg}")
            else:
                cast_reg = self.temp_state.new_tmp()
                lines.append(
                    f"  {cast_reg} = trunc {self.info.return_type} {call_reg} to i32"
                )
                lines.append(f"  ret i32 {cast_reg}")
        lines.append("}")
        return lines


class TempState:
    def __init__(self) -> None:
        self.counter = 0

    def new_tmp(self) -> str:
        self.counter += 1
        return f"%t{self.counter}"


def collect_var_types(function: Function) -> Dict[str, str]:
    var_types: Dict[str, str] = {}
    for arg in function.args:
        var_types[arg["name"]] = arg["type"]
    for instr in function.instrs:
        if isinstance(instr, (Const, ValueOperation)) and getattr(
            instr, "dest", None
        ):
            if instr.type is None:
                raise ValueError(
                    f"Missing type for instruction assigning to {instr.dest}"
                )
            var_types[instr.dest] = instr.type
    return var_types


def emit_instruction(
    instr: Instruction,
    lines: List[str],
    alloca_map: Dict[str, str],
    var_types: Dict[str, str],
    temp_state: TempState,
    name_map: Dict[str, str],
) -> bool:
    if isinstance(instr, Const):
        emit_const(instr, lines, alloca_map)
        return False
    if isinstance(instr, ValueOperation):
        return emit_value_op(
            instr, lines, alloca_map, var_types, temp_state, name_map
        )
    if isinstance(instr, EffectOperation):
        return emit_effect_op(
            instr, lines, alloca_map, var_types, temp_state, name_map
        )
    return False


def emit_const(instr: Const, lines: List[str], alloca_map: Dict[str, str]) -> None:
    dest = instr.dest
    llvm_ty = llvm_type(instr.type)
    value_repr = format_constant(instr.type, instr.value)
    ptr = alloca_map[dest]
    lines.append(f"  store {llvm_ty} {value_repr}, {llvm_ty}* {ptr}")


def emit_value_op(
    instr: ValueOperation,
    lines: List[str],
    alloca_map: Dict[str, str],
    var_types: Dict[str, str],
    temp_state: TempState,
    name_map: Dict[str, str],
) -> bool:
    op = instr.op
    if op == "id":
        src = load_value(instr.args[0], lines, alloca_map, var_types, temp_state)
        llvm_ty = llvm_type(instr.type)
        dest_ptr = alloca_map[instr.dest]
        lines.append(f"  store {llvm_ty} {src}, {llvm_ty}* {dest_ptr}")
        return False
    if op in {"add", "sub", "mul", "div"}:
        left = load_value(instr.args[0], lines, alloca_map, var_types, temp_state)
        right = load_value(instr.args[1], lines, alloca_map, var_types, temp_state)
        result = temp_state.new_tmp()
        llvm_ty = llvm_type(instr.type)
        opcode = {
            "add": "add nsw",
            "sub": "sub nsw",
            "mul": "mul nsw",
            "div": "sdiv",
        }[op]
        lines.append(f"  {result} = {opcode} {llvm_ty} {left}, {right}")
        dest_ptr = alloca_map[instr.dest]
        lines.append(f"  store {llvm_ty} {result}, {llvm_ty}* {dest_ptr}")
        return False
    if op in {"eq", "lt", "gt", "le", "ge", "ne"}:
        left_var = instr.args[0]
        right_var = instr.args[1]
        left = load_value(left_var, lines, alloca_map, var_types, temp_state)
        right = load_value(right_var, lines, alloca_map, var_types, temp_state)
        result = temp_state.new_tmp()
        operand_type = llvm_type(var_types[left_var])
        predicate = {
            "eq": "icmp eq",
            "lt": "icmp slt",
            "gt": "icmp sgt",
            "le": "icmp sle",
            "ge": "icmp sge",
            "ne": "icmp ne",
        }[op]
        lines.append(f"  {result} = {predicate} {operand_type} {left}, {right}")
        dest_ptr = alloca_map[instr.dest]
        lines.append(f"  store {LLVM_BOOL_TYPE} {result}, {LLVM_BOOL_TYPE}* {dest_ptr}")
        return False
    if op == "not":
        operand = load_value(instr.args[0], lines, alloca_map, var_types, temp_state)
        result = temp_state.new_tmp()
        lines.append(f"  {result} = xor i1 {operand}, true")
        dest_ptr = alloca_map[instr.dest]
        lines.append(f"  store i1 {result}, i1* {dest_ptr}")
        return False
    if op in {"and", "or"}:
        left = load_value(instr.args[0], lines, alloca_map, var_types, temp_state)
        right = load_value(instr.args[1], lines, alloca_map, var_types, temp_state)
        result = temp_state.new_tmp()
        opcode = "and" if op == "and" else "or"
        lines.append(f"  {result} = {opcode} i1 {left}, {right}")
        dest_ptr = alloca_map[instr.dest]
        lines.append(f"  store i1 {result}, i1* {dest_ptr}")
        return False
    if op == "call":
        callee = instr.funcs[0]
        callee_name = name_map.get(callee.lstrip("@"), f"@{callee.lstrip('@')}")
        call_args = []
        for arg in instr.args:
            arg_val = load_value(arg, lines, alloca_map, var_types, temp_state)
            arg_type = llvm_type(var_types[arg])
            call_args.append(f"{arg_type} {arg_val}")
        args_str = ", ".join(call_args)
        ret_ty = llvm_type(instr.type)
        if instr.dest:
            call_reg = temp_state.new_tmp()
            lines.append(f"  {call_reg} = call {ret_ty} {callee_name}({args_str})")
            dest_ptr = alloca_map[instr.dest]
            lines.append(f"  store {ret_ty} {call_reg}, {ret_ty}* {dest_ptr}")
        else:
            lines.append(f"  call {ret_ty} {callee_name}({args_str})")
        return False
    raise NotImplementedError(f"Unsupported value operation: {op}")


def emit_effect_op(
    instr: EffectOperation,
    lines: List[str],
    alloca_map: Dict[str, str],
    var_types: Dict[str, str],
    temp_state: TempState,
    name_map: Dict[str, str],
) -> bool:
    op = instr.op
    if op == "print":
        arg_count = len(instr.args)
        int_fmt = "getelementptr inbounds ([3 x i8], [3 x i8]* @.int_fmt, i32 0, i32 0)"
        space_ptr = "getelementptr inbounds ([2 x i8], [2 x i8]* @.space_str, i32 0, i32 0)"
        newline_ptr = "getelementptr inbounds ([2 x i8], [2 x i8]* @.newline_str, i32 0, i32 0)"
        true_ptr = "getelementptr inbounds ([5 x i8], [5 x i8]* @.true_str, i32 0, i32 0)"
        false_ptr = "getelementptr inbounds ([6 x i8], [6 x i8]* @.false_str, i32 0, i32 0)"
        for index, arg in enumerate(instr.args):
            arg_type = llvm_type(var_types[arg])
            value = load_value(arg, lines, alloca_map, var_types, temp_state)
            if arg_type == LLVM_INT_TYPE:
                lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {int_fmt}, {LLVM_INT_TYPE} {value})"
                )
            elif arg_type == LLVM_BOOL_TYPE:
                select_reg = temp_state.new_tmp()
                lines.append(
                    f"  {select_reg} = select i1 {value}, i8* {true_ptr}, i8* {false_ptr}"
                )
                lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {select_reg})"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported print type for variable {arg}: {arg_type}"
                )
            if index + 1 < arg_count:
                lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {space_ptr})"
                )
        lines.append(
            f"  call i32 (i8*, ...) @printf(i8* {newline_ptr})"
        )
        return False
    if op == "jmp":
        target = sanitize_label(instr.labels[0])
        lines.append(f"  br label %{target}")
        return True
    if op == "br":
        cond = load_value(instr.args[0], lines, alloca_map, var_types, temp_state)
        true_label = sanitize_label(instr.labels[0])
        false_label = sanitize_label(instr.labels[1])
        lines.append(
            f"  br i1 {cond}, label %{true_label}, label %{false_label}"
        )
        return True
    if op == "ret":
        if instr.args:
            value_name = instr.args[0]
            llvm_ty = llvm_type(var_types[value_name])
            value = load_value(value_name, lines, alloca_map, var_types, temp_state)
            lines.append(f"  ret {llvm_ty} {value}")
        else:
            lines.append("  ret void")
        return True
    if op == "call":
        callee = instr.funcs[0]
        callee_name = name_map.get(callee.lstrip("@"), f"@{callee.lstrip('@')}")
        call_args = []
        for arg in instr.args:
            arg_val = load_value(arg, lines, alloca_map, var_types, temp_state)
            arg_type = llvm_type(var_types[arg])
            call_args.append(f"{arg_type} {arg_val}")
        args_str = ", ".join(call_args)
        lines.append(f"  call void {callee_name}({args_str})")
        return False
    raise NotImplementedError(f"Unsupported effect operation: {op}")


def load_value(
    var: str,
    lines: List[str],
    alloca_map: Dict[str, str],
    var_types: Dict[str, str],
    temp_state: TempState,
) -> str:
    ptr = alloca_map[var]
    llvm_ty = llvm_type(var_types[var])
    temp = temp_state.new_tmp()
    lines.append(f"  {temp} = load {llvm_ty}, {llvm_ty}* {ptr}")
    return temp


def llvm_type(bril_type: Optional[str]) -> str:
    if bril_type is None or bril_type == "void":
        return "void"
    if bril_type == "int":
        return LLVM_INT_TYPE
    if bril_type == "bool":
        return LLVM_BOOL_TYPE
    raise NotImplementedError(f"Unsupported Bril type: {bril_type}")


def format_constant(bril_type: str, value) -> str:
    if bril_type == "int":
        return str(value)
    if bril_type == "bool":
        if isinstance(value, str):
            value = value.lower() == "true"
        return "1" if value else "0"
    raise NotImplementedError(f"Unsupported constant type: {bril_type}")


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"v_{sanitized}"
    return sanitized


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"label_{sanitized}"
    return sanitized
