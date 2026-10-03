# -*- coding: utf-8 -*-
"""
数据流分析：到达-定义（Reaching Definitions）与 使用-定义链（Use-Def Chain）。

回答"每个变量从定义到使用之间是怎么流动的"：
  * 定义点（def）—— var 声明 / 形参 / 赋值再定义 / 函数定义 / 内置函数；
  * 使用点（use）—— 表达式中读取标识符（含函数调用的被调名字、复合赋值左值的读）；
  * 对每个使用点，给出在该点"可能生效"的全部定义（到达该点的定义集合）。

分析完全建立在语义分析的符号解析结果之上（Identifier.symbol、VarDecl.symbol
已由 ``semantic.SemanticAnalyzer`` 绑定），因此：
  * 作用域遮蔽（shadowing）天然正确——内外层同名是不同的 Symbol，定义互不 KILL；
  * 未解析名字（symbol is None）单独标记，不臆造定义；
  * 多次赋值、循环中的再赋值、if/elif/else 分支汇合，由逐"区域"构建的
    控制流图（CFG）上的 may 数据流不动点迭代给出：
      IN[s]  = ⋃ OUT[p]（p 为前驱）
      OUT[s] = GEN[s] ∪ (IN[s] − KILL[s])
    循环靠回边（back edge）参与迭代直到不动点，于是循环体里的使用会同时到达
    "循环前的定义"与"循环体内上一轮的再定义"，体现真实的可能取值来源。

区域（region）：全局 <global> 与每个函数各一张 CFG。函数调用可读写全局变量，
故函数入口保守地播种全部全局定义（may 分析，宁可多报不漏报）。
"""

from . import ast_nodes as ast
from . import symbols as sym
from . import tokens as T


# 定义种类（中文标签由前端映射）
DEF_DECL = "decl"            # var 声明（带初值）
DEF_UNINIT = "uninit"        # var 声明（无初值，定义点即"未定义值"来源）
DEF_PARAM = "param"          # 形参
DEF_REASSIGN = "reassign"    # 赋值产生的再定义
DEF_FUNCTION = "function"    # 函数定义
DEF_BUILTIN = "builtin"      # 内置函数（合成定义）

# 使用上下文
USE_READ = "read"            # 普通取值
USE_UPDATE = "update"        # 复合赋值（+= 等）左值：先读后写
USE_CALLEE = "callee"        # 函数调用的被调名字
USE_WRITE = "write"          # 未解析名字上的写入（x = ...，x 未声明）

# CFG 边类型
EDGE_SEQ = "seq"
EDGE_TRUE = "true"
EDGE_FALSE = "false"
EDGE_BACK = "back"
EDGE_BREAK = "break"
EDGE_CONTINUE = "continue"


class Def:
    __slots__ = ("id", "symbol_id", "name", "kind", "line", "column", "length",
                 "region", "op", "type", "is_const")

    def __init__(self, did, symbol_id, name, kind, line, column, length,
                 region, op="", symbol_type="", is_const=False):
        self.id = did
        self.symbol_id = symbol_id
        self.name = name
        self.kind = kind
        self.line = line
        self.column = column
        self.length = length
        self.region = region
        self.op = op
        self.type = symbol_type
        self.is_const = is_const

    def to_dict(self):
        return {
            "id": self.id, "symbol_id": self.symbol_id, "name": self.name,
            "kind": self.kind, "line": self.line, "column": self.column,
            "length": self.length, "region": self.region, "op": self.op,
            "type": self.type, "is_const": self.is_const,
        }


class Use:
    __slots__ = ("id", "symbol_id", "name", "context", "line", "column",
                 "length", "region", "unresolved", "reaching", "stmt_kind")

    def __init__(self, uid, symbol_id, name, context, line, column, length,
                 region, unresolved=False, stmt_kind=""):
        self.id = uid
        self.symbol_id = symbol_id
        self.name = name
        self.context = context
        self.line = line
        self.column = column
        self.length = length
        self.region = region
        self.unresolved = unresolved
        self.reaching = []
        self.stmt_kind = stmt_kind

    def to_dict(self):
        return {
            "id": self.id, "symbol_id": self.symbol_id, "name": self.name,
            "context": self.context, "line": self.line, "column": self.column,
            "length": self.length, "region": self.region,
            "unresolved": self.unresolved,
            "reaching": list(self.reaching),
            "stmt_kind": self.stmt_kind,
        }


class Step:
    """CFG 节点：一条语句 / 条件判断 / 汇合锚点。"""
    __slots__ = ("index", "kind", "line", "uses", "gen", "kill", "preds", "succs",
                 "read_set", "out_set", "entry_seed")

    def __init__(self, index, kind, line=0):
        self.index = index
        self.kind = kind            # stmt | cond | anchor
        self.line = line
        self.uses = []
        self.gen = []               # List[Def.id]
        self.kill = []              # List[symbol_id]
        self.preds = []             # List[Step]
        self.succs = []             # List[(Step, edge_kind)]
        self.read_set = set()       # 读阶段可见定义（入口种子 ∪ 前驱 OUT）
        self.out_set = set()        # 写阶段产出（KILL 后 ∪ GEN），供后继
        self.entry_seed = set()     # 区域入口种子（仅入口步骤）


class _LoopCtx:
    """break / continue 需要前向引用循环结构，用栈暂挂其来源。"""
    __slots__ = ("break_srcs", "continue_srcs")

    def __init__(self):
        self.break_srcs = []        # List[Step]
        self.continue_srcs = []     # List[Step]


def _sym_id(symbol):
    return f"s{id(symbol)}"


# ---------------------------------------------------------------------------
# 表达式遍历：在已做符号解析的 AST 上收集 Identifier 读
# ---------------------------------------------------------------------------
def _child_exprs(node):
    """取出节点上所有直接表达式子节点（含列表/元组中的表达式）。"""
    out = []
    for v in vars(node).values():
        if isinstance(v, ast.Expr):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, ast.Expr):
                    out.append(x)
    return out


def collect_reads(expr, region, out, uid_holder, stmt_kind):
    """收集一个表达式中所有标识符读（Identifier 节点），追加为 Use。

    递归时带上"角色"上下文：函数被调名 -> callee，复合赋值左值 -> update。
    """
    def rec(e, ctx):
        if e is None:
            return
        if isinstance(e, ast.Identifier):
            s = getattr(e, "symbol", None)
            u = Use(f"u{uid_holder[0]}", _sym_id(s) if s else None, e.name,
                    ctx, e.line, e.column, len(e.name), region,
                    unresolved=s is None, stmt_kind=stmt_kind)
            uid_holder[0] += 1
            out.append(u)
            return
        if isinstance(e, ast.CallExpr):
            if isinstance(e.callee, ast.Identifier):
                rec(e.callee, USE_CALLEE)
            else:
                rec(e.callee, USE_READ)
            for a in e.args:
                rec(a, USE_READ)
            return
        if isinstance(e, ast.IndexExpr):
            rec(e.target, USE_READ)
            rec(e.index, USE_READ)
            return
        if isinstance(e, ast.AssignStmt):
            # 防御：赋值表达式出现在表达式位置（如 for 增量已在语句层处理）
            rec(e.value, USE_READ)
            if isinstance(e.target, ast.Identifier):
                if e.op != "=":
                    rec(e.target, USE_UPDATE)
            else:
                rec(e.target, USE_READ)
            return
        for c in _child_exprs(e):
            rec(c, USE_READ)

    rec(expr, USE_READ)


# ---------------------------------------------------------------------------
# 名字记号定位（语义 AST 给的是 var/func 关键字位置，名字列从 token 流取）
# ---------------------------------------------------------------------------
def _name_token_after(tokens, line, column, kw_type):
    for i, tok in enumerate(tokens):
        if tok.type == kw_type and tok.line == line and tok.column == column:
            for t in tokens[i + 1:]:
                if t.type == T.IDENT:
                    return t
                if t.type in (T.SEMICOLON, T.LBRACE, T.EOF, T.LPAREN):
                    break
    return None


# ---------------------------------------------------------------------------
# 区域 CFG 构建器
# ---------------------------------------------------------------------------
class _RegionBuilder:
    def __init__(self, region, analyzer):
        self.region = region
        self.a = analyzer
        self.steps = []
        self.edges = []                 # [(from_idx, to_idx, kind)]
        self.loop_stack = []

    def _emit(self, kind, line=0):
        s = Step(len(self.steps), kind, line)
        self.steps.append(s)
        return s

    def _edge(self, src, dst, kind):
        if src is None or dst is None:
            return
        src.succs.append((dst, kind))
        dst.preds.append(src)
        self.edges.append((src.index, dst.index, kind))

    def _def(self, symbol, name, kind, line, column, length, op=""):
        return self.a.make_def(self.region, symbol, name, kind,
                               line, column, length, op)

    # ---- 语句链：[(src_step, edge_kind)] 入口 -> [(src_step, edge_kind)] 出口 ----
    def build_stmts(self, stmts, entries):
        exits = list(entries)
        for s in stmts:
            if s is None:
                continue
            exits = self.build_stmt(s, exits)
        return exits

    def build_stmt(self, stmt, entries):
        if isinstance(stmt, ast.Block):
            return self.build_stmts(stmt.statements, entries)

        if isinstance(stmt, ast.VarDecl):
            step = self._emit("stmt", stmt.line)
            for p, k in entries:
                self._edge(p, step, k)
            if stmt.initializer is not None:
                collect_reads(stmt.initializer, self.region, step.uses,
                              self.a.uid, "VarDecl")
            symbol = getattr(stmt, "symbol", None)
            tok = getattr(stmt, "_name_token", None)
            col = tok.column if tok else stmt.column
            length = len(tok.text) if tok else len(stmt.name)
            kind = DEF_DECL if stmt.initializer is not None else DEF_UNINIT
            d = self._def(symbol, stmt.name, kind, stmt.line, col, length)
            step.gen.append(d.id)
            if symbol is not None:
                step.kill.append(_sym_id(symbol))
            return [(step, EDGE_SEQ)]

        assign = stmt if isinstance(stmt, ast.AssignStmt) else None
        if assign is None and isinstance(stmt, ast.ExprStmt) and isinstance(stmt.expr, ast.AssignStmt):
            assign = stmt.expr
        if assign is not None:
            return [(self._build_assign(assign, entries, "AssignStmt"), EDGE_SEQ)]

        if isinstance(stmt, ast.PrintStmt):
            step = self._emit("stmt", stmt.line)
            for p, k in entries:
                self._edge(p, step, k)
            for x in stmt.args:
                collect_reads(x, self.region, step.uses, self.a.uid, "PrintStmt")
            return [(step, EDGE_SEQ)]

        if isinstance(stmt, ast.ExprStmt):
            step = self._emit("stmt", stmt.line)
            for p, k in entries:
                self._edge(p, step, k)
            collect_reads(stmt.expr, self.region, step.uses, self.a.uid, "ExprStmt")
            return [(step, EDGE_SEQ)]

        if isinstance(stmt, ast.IfStmt):
            return self._build_if(stmt, entries)
        if isinstance(stmt, ast.WhileStmt):
            return self._build_while(stmt, entries)
        if isinstance(stmt, ast.ForStmt):
            return self._build_for(stmt, entries)

        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is not None:
                step = self._emit("stmt", stmt.line)
                for p, k in entries:
                    self._edge(p, step, k)
                collect_reads(stmt.value, self.region, step.uses,
                              self.a.uid, "ReturnStmt")
            return []   # 终结

        if isinstance(stmt, (ast.BreakStmt, ast.ContinueStmt)):
            ctx = self.loop_stack[-1] if self.loop_stack else None
            if ctx is not None:
                if isinstance(stmt, ast.BreakStmt):
                    ctx.break_srcs.extend(entries)
                else:
                    ctx.continue_srcs.extend(entries)
            return []

        if isinstance(stmt, ast.FunctionDecl):
            # 嵌套函数提升到区域入口处理；语句位置不产生顺序流
            return entries

        return entries

    def _build_assign(self, assign, entries, stmt_kind):
        step = self._emit("stmt", assign.line)
        for p, k in entries:
            self._edge(p, step, k)
        collect_reads(assign.value, self.region, step.uses, self.a.uid, stmt_kind)
        target = assign.target
        if isinstance(target, ast.Identifier):
            symbol = getattr(target, "symbol", None)
            if symbol is not None and symbol.kind != sym.KIND_BUILTIN:
                if assign.op != "=":
                    # 复合赋值左值先读（update 角色）
                    collect_reads(target, self.region, step.uses, self.a.uid, stmt_kind)
                d = self._def(symbol, target.name, DEF_REASSIGN,
                              target.line, target.column, len(target.name),
                              op=assign.op)
                step.gen.append(d.id)
                step.kill.append(_sym_id(symbol))
            else:
                # 未声明名字的写入：未解析使用，不产生定义
                u = Use(f"u{self.a.uid[0]}", None, target.name, USE_WRITE,
                        target.line, target.column, len(target.name),
                        self.region, unresolved=True, stmt_kind=stmt_kind)
                self.a.uid[0] += 1
                step.uses.append(u)
        else:
            # 下标赋值 a[i] = v：a、i 都是读（target 遍历已含），无新定义
            collect_reads(target, self.region, step.uses, self.a.uid, stmt_kind)
        return step

    def _build_if(self, stmt, entries):
        anchor = self._emit("anchor", stmt.line)
        cond_steps = []
        for cond, body in stmt.branches:
            cstep = self._emit("cond", cond.line)
            cond_steps.append(cstep)
            collect_reads(cond, self.region, cstep.uses, self.a.uid, "condition")
        # 外部入口 -> 首条件（保留边类型）
        for p, k in entries:
            self._edge(p, cond_steps[0], k)
        # elif：前一条件为假进入下一条件
        for a, b in zip(cond_steps, cond_steps[1:]):
            self._edge(a, b, EDGE_FALSE)
        last_cond = cond_steps[-1]
        branch_exits = []
        for (cond, body), cstep in zip(stmt.branches, cond_steps):
            exits = self.build_stmts(body.statements, [(cstep, EDGE_TRUE)])
            branch_exits.extend(exits)
        if stmt.else_block is not None:
            exits = self.build_stmts(stmt.else_block.statements,
                                     [(last_cond, EDGE_FALSE)])
            branch_exits.extend(exits)
        else:
            branch_exits.append((last_cond, EDGE_FALSE))
        for e, k in branch_exits:
            self._edge(e, anchor, k)
        return [(anchor, EDGE_SEQ)]

    def _build_while(self, stmt, entries):
        ctx = _LoopCtx()
        self.loop_stack.append(ctx)
        cond = self._emit("cond", stmt.condition.line)
        for p, k in entries:
            self._edge(p, cond, k)
        collect_reads(stmt.condition, self.region, cond.uses, self.a.uid, "condition")
        body_exits = self.build_stmts(stmt.body.statements, [(cond, EDGE_TRUE)])
        anchor = self._emit("anchor", stmt.line)
        for e, _k in body_exits:
            self._edge(e, cond, EDGE_BACK)
        for src, _k in ctx.continue_srcs:
            self._edge(src, cond, EDGE_CONTINUE)
        self._edge(cond, anchor, EDGE_FALSE)
        for src, k in ctx.break_srcs:
            self._edge(src, anchor, k)
        self.loop_stack.pop()
        return [(anchor, EDGE_SEQ)]

    def _build_for(self, stmt, entries):
        ctx = _LoopCtx()
        self.loop_stack.append(ctx)
        exits = list(entries)
        if stmt.init is not None:
            exits = self.build_stmt(stmt.init, exits)

        cond = None
        if stmt.condition is not None:
            cond = self._emit("cond", stmt.condition.line)
            for p, k in exits:
                self._edge(p, cond, k)
            collect_reads(stmt.condition, self.region, cond.uses,
                          self.a.uid, "condition")
            body_entries = [(cond, EDGE_TRUE)]
        else:
            body_entries = [(p, k) for p, k in exits]

        body_exits = self.build_stmts(stmt.body.statements, body_entries)

        # 增量：通常为赋值
        if stmt.increment is not None:
            if isinstance(stmt.increment, ast.AssignStmt):
                inc = self._build_assign(stmt.increment, body_exits, "ForIncrement")
            else:
                inc = self._emit("stmt", stmt.increment.line)
                for p, k in body_exits:
                    self._edge(p, inc, k)
                collect_reads(stmt.increment, self.region, inc.uses,
                              self.a.uid, "ForIncrement")
            inc_exits = [(inc, EDGE_SEQ)]
        else:
            inc_exits = body_exits

        anchor = self._emit("anchor", stmt.line)
        if cond is not None:
            for e, _k in inc_exits:
                self._edge(e, cond, EDGE_BACK)
            for src, _k in ctx.continue_srcs:
                target = inc if stmt.increment is not None else cond
                self._edge(src, target, EDGE_CONTINUE)
            self._edge(cond, anchor, EDGE_FALSE)
        else:
            # 无条件循环：增量后回到循环体入口，只能经 break 退出
            first = body_entries[0][0] if body_entries else None
            for e, _k in inc_exits:
                if first is not None:
                    self._edge(e, first, EDGE_BACK)
            for src, _k in ctx.continue_srcs:
                target = inc if stmt.increment is not None else first
                if target is not None:
                    self._edge(src, target, EDGE_CONTINUE)
        for src, k in ctx.break_srcs:
            self._edge(src, anchor, k)
        self.loop_stack.pop()
        return [(anchor, EDGE_SEQ)]


# ---------------------------------------------------------------------------
# 主分析器
# ---------------------------------------------------------------------------
class DataFlowAnalyzer:
    def __init__(self, program, symbol_table, tokens):
        self.program = program
        self.symbols = symbol_table
        self.tokens = tokens
        self.uid = [0]
        self._dnum = 0
        self.all_defs = []
        self.all_uses = []
        self.regions = []
        self.symbol_info = {}
        self.builtin_defs = []
        self.function_defs = []
        self.global_var_defs = []

    def make_def(self, region, symbol, name, kind, line, column, length, op=""):
        self._dnum += 1
        d = Def(f"d{self._dnum}", _sym_id(symbol) if symbol else None,
                name, kind, line, column, length, region, op=op,
                symbol_type=getattr(symbol, "symbol_type", "") if symbol else "",
                is_const=getattr(symbol, "is_const", False) if symbol else False)
        self.all_defs.append(d)
        if symbol is not None:
            self.register_symbol(symbol)
        return d

    def register_symbol(self, s):
        sid = _sym_id(s)
        if sid not in self.symbol_info:
            self.symbol_info[sid] = {
                "id": sid, "name": s.name, "kind": s.kind,
                "type": s.symbol_type, "scope_id": s.scope.scope_id,
                "scope_type": s.scope.scope_type, "scope_name": s.scope.name,
                "line": s.line, "column": s.column, "is_const": s.is_const,
            }
        return sid

    # ---- AST 遍历工具 ----
    def _walk_nodes(self, node):
        if isinstance(node, ast.Node):
            yield node
        for v in vars(node).values():
            if isinstance(v, ast.Node):
                yield from self._walk_nodes(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, ast.Node):
                        yield from self._walk_nodes(x)
            elif isinstance(v, tuple):
                for x in v:
                    if isinstance(x, ast.Node):
                        yield from self._walk_nodes(x)

    def _inject_name_tokens(self):
        for n in self._walk_nodes(self.program):
            if isinstance(n, ast.VarDecl):
                n._name_token = _name_token_after(self.tokens, n.line, n.column, T.KW_VAR)
            elif isinstance(n, ast.FunctionDecl):
                n._name_token = _name_token_after(self.tokens, n.line, n.column, T.KW_FUNC)

    # ---- 入口 ----
    def analyze(self):
        self._inject_name_tokens()
        self._make_builtin_defs()

        # 预先定位每个函数定义"所属区域"（顶层函数 -> <global>，嵌套 -> 外层函数）
        self._function_regions = self._map_function_regions()

        # 先为所有区域创建函数名的提升定义（保证嵌套函数在外层区域求解时已可见）
        for region in self._all_region_names():
            self._hoisted_function_defs(region)

        # 1) 全局区域
        seed = list(self.builtin_defs) + [d for d in self.function_defs
                                          if d.region == "<global>"]
        global_stmts = [d for d in self.program.declarations if isinstance(d, ast.Stmt)]
        self._solve_region("<global>", "global", global_stmts, seed, line=1)

        # 收集全局变量/形参符号的全部定义，作为函数入口的保守种子
        gscope = self.symbols.global_scope
        global_sids = {_sym_id(s) for s in gscope.symbols.values()
                       if s.kind in (sym.KIND_VARIABLE, sym.KIND_PARAMETER)}
        for d in self.all_defs:
            if d.symbol_id in global_sids:
                self.global_var_defs.append(d)

        # 2) 每个函数（含嵌套）一个区域，按其出现顺序（外层先于内层）求解
        for fn in self._all_functions():
            self._solve_function(fn)

        return self._result()

    def _all_functions(self):
        return [n for n in self._walk_nodes(self.program)
                if isinstance(n, ast.FunctionDecl)]

    def _map_function_regions(self):
        """返回 {id(FunctionDecl): region_name}。

        函数声明要么直接挂在 Program.declarations，要么位于某个函数体中
        （顶层语句 / if / 循环 / 嵌套块都算该函数区域）。
        """
        result = {}

        def scan_stmts(stmts, region):
            for st in stmts:
                _scan_stmt(st, region)

        def _scan_stmt(node, region):
            if node is None:
                return
            if isinstance(node, ast.FunctionDecl):
                result[id(node)] = region
                # 函数体内的声明属于该新函数自己的区域
                scan_stmts(node.body.statements, node.name)
                return
            if isinstance(node, ast.Block):
                scan_stmts(node.statements, region)
            elif isinstance(node, ast.IfStmt):
                for _c, body in node.branches:
                    scan_stmts(body.statements, region)
                if node.else_block:
                    scan_stmts(node.else_block.statements, region)
            elif isinstance(node, ast.WhileStmt):
                scan_stmts(node.body.statements, region)
            elif isinstance(node, ast.ForStmt):
                if node.init:
                    _scan_stmt(node.init, region)
                scan_stmts(node.body.statements, region)

        for decl in self.program.declarations:
            _scan_stmt(decl, "<global>")
        return result

    def _make_builtin_defs(self):
        gscope = self.symbols.global_scope
        for name in sorted(gscope.symbols):
            s = gscope.symbols[name]
            if s.kind != sym.KIND_BUILTIN:
                continue
            self.register_symbol(s)
            d = self.make_def("<global>", s, name, DEF_BUILTIN, 0, 0, len(name))
            self.builtin_defs.append(d)

    def _all_region_names(self):
        names = ["<global>"]
        for fn in self._all_functions():
            r = self._function_regions.get(id(fn), fn.name)
            # 函数自身的区域名就是函数名（即使嵌套）
            if fn.name not in names:
                names.append(fn.name)
        return names

    def _hoisted_function_defs(self, region):
        """为声明在该区域的所有函数名创建提升定义（幂等，已创建则跳过）。"""
        seed = list(self.builtin_defs) if region == "<global>" else []
        for fn in self._all_functions():
            if self._function_regions.get(id(fn)) != region:
                continue
            existing = next((d for d in self.function_defs
                             if d.region == region and d.name == fn.name
                             and d.line == fn.line), None)
            if existing is not None:
                seed.append(existing)
                continue
            s = getattr(fn, "symbol", None)
            tok = getattr(fn, "_name_token", None)
            line = tok.line if tok else fn.line
            col = tok.column if tok else fn.column
            length = len(tok.text) if tok else len(fn.name)
            d = self.make_def(region, s, fn.name, DEF_FUNCTION,
                              line, col, length)
            self.function_defs.append(d)
            seed.append(d)
        return seed

    def _solve_function(self, fn):
        region = self._function_regions.get(id(fn), fn.name)
        seed = self._region_seed_defs(fn, region)
        # fn.symbol 挂在全局作用域，形参所在的是函数体对应的 SCOPE_FUNCTION 作用域
        scope = self._find_function_scope(fn)
        param_toks = self._param_tokens(fn)
        for i, pname in enumerate(fn.params):
            psym = scope.lookup_local(pname) if scope is not None else None
            tok = param_toks[i] if i < len(param_toks) else None
            line = tok.line if tok else fn.line
            col = tok.column if tok else fn.column
            length = len(tok.text) if tok else len(pname)
            d = self.make_def(fn.name, psym, pname, DEF_PARAM, line, col, length)
            seed.append(d)
        # 保守：函数可能在任意全局定义之后被调用，播种全部全局变量定义
        seed.extend(self.global_var_defs)
        self._solve_region(fn.name, "function", fn.body.statements, seed,
                           line=fn.line)

    def _region_seed_defs(self, fn, region):
        """函数入口可见的函数定义与外层捕获变量：
          * 全部内置函数；
          * 声明在本区域或全局区域的函数（递归 / 互调 / 词法捕获）；
          * 沿作用域链向外可见的变量与形参（闭包捕获，如嵌套函数读外层 x）。
        到达解析按符号过滤，跨作用域同名符号不会错误串线（遮蔽天然隔离）。
        """
        seed = list(self.builtin_defs)
        for d in self.function_defs:
            if d.region in (region, "<global>"):
                seed.append(d)
        # 外层作用域可见的变量/形参：用该函数 SCOPE_FUNCTION 的 parent 链
        fscope = self._find_function_scope(fn)
        captured_sids = set()
        sc = fscope.parent if fscope is not None else None
        while sc is not None:
            for s in sc.symbols.values():
                if s.kind in (sym.KIND_VARIABLE, sym.KIND_PARAMETER):
                    captured_sids.add(_sym_id(s))
            sc = sc.parent
        if captured_sids:
            for d in self.all_defs:
                if d.symbol_id in captured_sids:
                    seed.append(d)
        return seed

    def _find_function_scope(self, fn):
        """在作用域树中找函数名对应的 SCOPE_FUNCTION 作用域。"""
        for sc in self.symbols.scopes:
            if sc.scope_type == sym.SCOPE_FUNCTION and sc.name == fn.name:
                return sc
        return None

    def _param_tokens(self, fn):
        start = None
        for i, tok in enumerate(self.tokens):
            if tok.type == T.KW_FUNC and tok.line == fn.line and tok.column == fn.column:
                start = i
                break
        if start is None:
            return []
        out, depth = [], 0
        for t in self.tokens[start:]:
            if t.type == T.LPAREN:
                depth += 1
                continue
            if depth == 1:
                if t.type == T.RPAREN:
                    break
                if t.type == T.IDENT and t.text != fn.name:
                    out.append(t)
        return out[:len(fn.params)]

    # ---- 构建 CFG + 不动点 ----
    def _solve_region(self, name, kind, stmts, seed, line=0):
        builder = _RegionBuilder(name, self)
        builder.build_stmts(stmts, [])
        steps = builder.steps

        seed_ids = {d.id for d in seed}
        entries = [s for s in steps if not s.preds]
        for s in entries:
            s.entry_seed = set(seed_ids)

        # 构建过程中还会新建本区域定义（var 声明等），每次迭代前刷新 id -> 符号映射
        def refresh():
            return {d.id: d.symbol_id for d in self.all_defs}

        # 初始化：入口读集合用种子，所有写集合（OUT）留空——首轮按 CFG 顺序传播。
        for s in entries:
            s.read_set = set(seed_ids)
            s.out_set = set()

        for _ in range(500):
            did_sid = refresh()
            changed = False
            for s in steps:
                # 读阶段：入口种子 ∪ 前驱 OUT（语句内部读据此解析）
                new_read = set(s.entry_seed)
                for p in s.preds:
                    new_read |= p.out_set
                # 写阶段：KILL 同符号旧定义后加入本步 GEN，供后继使用
                new_out = set(new_read)
                for sym_id in s.kill:
                    new_out = {did for did in new_out
                               if did_sid.get(did) != sym_id}
                new_out |= set(s.gen)
                if new_read != s.read_set or new_out != s.out_set:
                    s.read_set, s.out_set = new_read, new_out
                    changed = True
            if not changed:
                break

        did_sid = refresh()

        # 使用点解析：语句内部读（RHS/初值/复合左值/条件）看"读阶段"集合。
        # 回边带来的是上一轮执行产生的定义，故循环中再赋值的使用会同时到达
        # 循环前定义与上一轮的再定义；本步自己的 GEN 只出现在 OUT，不会自指。
        region_uses = []
        for s in steps:
            for u in s.uses:
                if u.symbol_id is not None:
                    ids = [did for did in s.read_set
                           if did_sid.get(did) == u.symbol_id]
                    ids.sort(key=lambda x: int(x[1:]))
                    u.reaching = ids
                self.all_uses.append(u)
                region_uses.append(u)

        self.regions.append({
            "name": name, "kind": kind, "line": line,
            "steps": len(steps),
            "edges": len(builder.edges),
            "def_count": sum(len(st.gen) for st in steps),
            "use_count": len(region_uses),
            "cfg": {"edges": [{"from": a, "to": b, "kind": k}
                              for a, b, k in builder.edges],
                    "steps": [{"index": st.index, "kind": st.kind,
                               "line": st.line,
                               "gen": list(st.gen),
                               "uses": [u.id for u in st.uses]}
                              for st in steps]},
        })

    def _result(self):
        merged = sum(1 for u in self.all_uses if len(u.reaching) > 1)
        unresolved = sum(1 for u in self.all_uses if u.unresolved)
        uninit_ids = {d.id for d in self.all_defs if d.kind == DEF_UNINIT}
        uninit = sum(1 for u in self.all_uses
                     if any(did in uninit_ids for did in u.reaching))
        return {
            "available": True,
            "symbols": list(self.symbol_info.values()),
            "defs": [d.to_dict() for d in self.all_defs],
            "uses": [u.to_dict() for u in self.all_uses],
            "regions": [{k: v for k, v in r.items() if k != "cfg"} for r in self.regions],
            "cfgs": {r["name"]: r["cfg"] for r in self.regions},
            "stats": {
                "symbol_count": len(self.symbol_info),
                "def_count": len(self.all_defs),
                "use_count": len(self.all_uses),
                "merged_use_count": merged,
                "unresolved_count": unresolved,
                "uninit_use_count": uninit,
            },
        }


def analyze(program, symbol_table, tokens):
    """对外便捷入口。"""
    return DataFlowAnalyzer(program, symbol_table, tokens).analyze()
