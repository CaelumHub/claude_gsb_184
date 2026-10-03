# -*- coding: utf-8 -*-
"""
数据流分析：到达-定义（Reaching Definitions）。

在语义分析产出的符号解析结果（``Identifier.symbol`` / ``VarDecl.symbol``）
之上，计算每个**使用点**真正读取的是哪一个（哪些）**定义点**写入的值，
正确覆盖：

* 多次赋值 —— 每次赋值都是一个独立定义点，后定义沿顺序流"杀死"前定义；
* 分支汇合 —— 不同分支里的定义在汇合点同时到达（使用点可能对应多个定义）；
* 循环再赋值 —— 沿回边迭代到不动点：循环体里的再定义可以到达循环条件
  与循环体内后续迭代中的使用点（back-edge）；
* 作用域遮蔽 —— 遮蔽变量在语义分析阶段就是另一个 Symbol，天然与外层
  变量互不影响；
* 形参 / 函数名 / 内置函数 —— 分别是函数入口、全局与"内置"伪定义点。

分析方法（与经典编译原理一致）：
  gen/b 集合：gen 为语句产生的定义，kill 为被它覆盖的同一 Symbol 的其它定义；
  顺序流传递当前环境（symbol -> 到达定义集合），分支做并集汇合，
  循环在"循环头环境"上迭代直到环境不再变化（不动点）。

输出按"区域"（<main> 与每个函数）组织的 defs / uses / flows，供
"数据流分析"页面以连线或列表呈现。
"""

from . import ast_nodes as ast
from . import symbols as sym
from . import semantic as semantic_mod
from . import lexer as lexer_mod
from . import parser as parser_mod
from . import diagnostics as diag

# 定义点种类（中文标签在前端映射）
KIND_DECL = "decl"            # var 声明（含初始化）
KIND_ASSIGN = "assign"        # 普通再赋值 x = ...
KIND_COMPOUND = "compound"    # 复合赋值 x += ...（既是定义又是使用）
KIND_PARAM = "parameter"      # 形参（函数入口定义）
KIND_FUNCTION = "function"    # 函数声明（全局定义点）
KIND_BUILTIN = "builtin"      # 内置函数伪定义点
KIND_UNDEFINED = "undefined"  # 对未定义名字的赋值（解析不到 Symbol）


class _Def:
    __slots__ = ("id", "kind", "name", "line", "column", "end_column",
                 "symbol", "scope_type", "scope_name", "region", "is_const")

    def __init__(self, did, kind, name, line, column, end_column, symbol,
                 scope_type, scope_name, region, is_const=False):
        self.id = did
        self.kind = kind
        self.name = name
        self.line = line
        self.column = column          # 1-based，含末列（左闭右闭）
        self.end_column = end_column
        self.symbol = symbol          # 语义分析绑定的 Symbol；未定义名字为 None
        self.scope_type = scope_type
        self.scope_name = scope_name
        self.region = region          # 所属 _Region
        self.is_const = is_const


class _Use:
    __slots__ = ("id", "name", "line", "column", "end_column", "symbol",
                 "region", "context", "reaching", "status")

    def __init__(self, uid, name, line, column, end_column, symbol, region, context):
        self.id = uid
        self.name = name
        self.line = line
        self.column = column
        self.end_column = end_column
        self.symbol = symbol
        self.region = region
        self.context = context        # read | call | compound
        self.reaching = []            # List[_Def]，最终填充
        self.status = "ok"            # ok | merged | uninit | undefined


class _Region:
    """一个被分析的控制流区域：<main> 或一个函数体。"""

    def __init__(self, key, name, node, scope, is_main=False):
        self.key = key                # main | 函数名
        self.name = name
        self.node = node              # Program（main）或 FunctionDecl
        self.scope = scope
        self.is_main = is_main
        self.defs = []                # 本区域词法上出现的定义点
        self.uses = []                # 本区域词法上出现的使用点
        self.entry_params = []        # 形参定义点（仅函数）
        self.nested_func_defs = []    # 区域内嵌套函数声明产生的定义点


class DataFlowAnalyzer:
    def __init__(self, source):
        self.regions = []
        self.region_by_node = {}
        self._seq = 0
        self._symbol_defs = {}        # Symbol id(_Symbol) -> 所有定义点（全程序）

    # ------------------------------------------------------------------
    # id
    # ------------------------------------------------------------------
    def _did(self):
        self._seq += 1
        return f"d{self._seq}"

    def _uid(self):
        self._seq += 1
        return f"u{self._seq}"

    def _scope_label(self, scope):
        return scope.scope_type, (scope.name if scope else "")

    # ------------------------------------------------------------------
    # 第一阶段：建区域 + 结构化收集定义点/使用点（此时不做控制流）
    # ------------------------------------------------------------------
    def _collect(self, program, table):
        # main 区域
        main_scope = table.global_scope
        main_region = _Region("main", "<main>", program, main_scope, is_main=True)
        self.regions.append(main_region)
        self.region_by_node[program] = main_region

        # 先为所有函数建区域（函数名已在语义第一遍注册进全局作用域）
        for decl in program.declarations:
            if isinstance(decl, ast.FunctionDecl):
                r = _Region(decl.name, decl.name, decl, decl.symbol.scope if decl.symbol else main_scope)
                self.regions.append(r)
                self.region_by_node[decl] = r

        # main 中的语句（函数声明作为 main 区域的定义点，函数体跳过）
        for d in program.declarations:
            if isinstance(d, ast.FunctionDecl):
                self._collect_func_name(main_region, d)
            else:
                self._collect_stmt(d, main_region)

        # 预声明、但词法上没有 var 定义点的顶层变量（例如源码中只有函数
        # 引用、却未声明的全局名由语义层兜底的情形）——通常不会出现，
        # 这里保守地为其补一个定义点，保证函数区域的引用有可达来源。
        for name, s in main_scope.symbols.items():
            if s.kind != sym.KIND_VARIABLE:
                continue
            if any(defn.symbol is s for defn in main_region.defs):
                continue
            d = _Def(self._did(), KIND_DECL, name, 1, 1, 1 + len(name), s,
                     sym.SCOPE_GLOBAL, "global", main_region)
            main_region.defs.append(d)
            self._symbol_defs.setdefault(id(s), []).append(d)

        # 各函数：形参入口定义点 + 函数体
        for r in self.regions:
            if r.is_main:
                continue
            fn = r.node
            for i, p in enumerate(fn.params):
                psym = (fn.param_symbols or [None] * len(fn.params))[i] \
                    if getattr(fn, "param_symbols", None) else None
                col = (fn.param_columns or {}).get(i, fn.column)
                d = _Def(self._did(), KIND_PARAM, p, fn.line, col, col + len(p),
                         psym, sym.SCOPE_FUNCTION, fn.name, r)
                r.entry_params.append(d)
                r.defs.append(d)
                if psym is not None:
                    self._symbol_defs.setdefault(id(psym), []).append(d)
            self._collect_block(fn.body, r)

    def _collect_func_name(self, region, fn):
        st, sn = self._scope_label(region.scope)
        col = getattr(fn, "name_column", fn.column)
        d = _Def(self._did(), KIND_FUNCTION, fn.name, fn.line, col,
                 col + len(fn.name), fn.symbol, st, sn, region)
        region.defs.append(d)
        region.nested_func_defs.append(d)
        if fn.symbol is not None:
            self._symbol_defs.setdefault(id(fn.symbol), []).append(d)

    def _collect_block(self, block, region):
        for s in block.statements:
            self._collect_stmt(s, region)

    def _collect_stmt(self, s, region):
        if s is None:
            return
        if isinstance(s, ast.Block):
            self._collect_block(s, region)
        elif isinstance(s, ast.VarDecl):
            symbol = s.symbol
            st, sn = self._scope_label(symbol.scope if symbol else region.scope)
            col = getattr(s, "name_column", s.column)
            d = _Def(self._did(), KIND_DECL, s.name, s.line, col, col + len(s.name),
                     symbol, st, sn, region, is_const=s.is_const)
            region.defs.append(d)
            if symbol is not None:
                self._symbol_defs.setdefault(id(symbol), []).append(d)
            if s.initializer:
                self._collect_expr(s.initializer, region)
        elif isinstance(s, ast.AssignStmt):
            self._collect_assign(s, region)
        elif isinstance(s, ast.ExprStmt):
            self._collect_expr(s.expr, region)
        elif isinstance(s, ast.PrintStmt):
            # print 本身也是内置函数 print 的使用点（语句形式）
            self._mk_builtin_use("print", s, region)
            for a in s.args:
                self._collect_expr(a, region)
        elif isinstance(s, ast.IfStmt):
            for cond, body in s.branches:
                self._collect_expr(cond, region)
                self._collect_block(body, region)
            if s.else_block:
                self._collect_block(s.else_block, region)
        elif isinstance(s, ast.WhileStmt):
            self._collect_expr(s.condition, region)
            self._collect_block(s.body, region)
        elif isinstance(s, ast.ForStmt):
            if s.init:
                self._collect_stmt(s.init, region)
            if s.condition:
                self._collect_expr(s.condition, region)
            if s.increment:
                # 增量通常是赋值语句（解析器允许它写成表达式语句形式）
                if isinstance(s.increment, ast.AssignStmt):
                    self._collect_assign(s.increment, region)
                else:
                    self._collect_expr(s.increment, region)
            self._collect_block(s.body, region)
        elif isinstance(s, ast.ReturnStmt):
            if s.value:
                self._collect_expr(s.value, region)
        elif isinstance(s, ast.FunctionDecl):
            # 嵌套函数声明：在当前区域是一个定义点；函数体是独立区域
            self._collect_func_name(region, s)
        # Break / Continue：无变量读写

    def _collect_assign(self, s, region):
        # 复合赋值的左值本身先被读取（x += 1 等价 x = x + 1）
        if s.op != "=" and isinstance(s.target, ast.Identifier):
            self._mk_use(s.target, region, KIND_COMPOUND)
        if isinstance(s.target, ast.Identifier):
            symbol = s.target.symbol
            kind = KIND_ASSIGN if s.op == "=" else KIND_COMPOUND
            if symbol is None:
                kind = KIND_UNDEFINED
            st, sn = self._scope_label(symbol.scope if symbol else region.scope)
            d = _Def(self._did(), kind, s.target.name, s.target.line, s.target.column,
                     s.target.column + len(s.target.name), symbol, st, sn, region)
            region.defs.append(d)
            if symbol is not None:
                self._symbol_defs.setdefault(id(symbol), []).append(d)
        elif isinstance(s.target, ast.IndexExpr):
            # a[i] = v：不是 a 的新定义（只产生 a、i、v 的使用）
            self._collect_expr(s.target.target, region)
            self._collect_expr(s.target.index, region)
        self._collect_expr(s.value, region)

    def _collect_expr(self, e, region):
        if e is None:
            return
        if isinstance(e, ast.Identifier):
            self._mk_use(e, region, "read")
        elif isinstance(e, ast.AssignStmt):
            # 表达式位置出现的赋值（如 for 增量 / 短路表达式内）
            self._collect_assign(e, region)
        elif isinstance(e, (ast.BinaryExpr, ast.LogicalExpr)):
            self._collect_expr(e.left, region)
            self._collect_expr(e.right, region)
        elif isinstance(e, ast.UnaryExpr):
            self._collect_expr(e.operand, region)
        elif isinstance(e, ast.CallExpr):
            if isinstance(e.callee, ast.Identifier):
                self._mk_use(e.callee, region, "call")
            else:
                self._collect_expr(e.callee, region)
            for a in e.args:
                self._collect_expr(a, region)
        elif isinstance(e, ast.IndexExpr):
            self._collect_expr(e.target, region)
            self._collect_expr(e.index, region)
        elif isinstance(e, ast.ListLiteral):
            for x in e.elements:
                self._collect_expr(x, region)
        # 字面量无使用

    def _mk_use(self, ident, region, context):
        u = _Use(self._uid(), ident.name, ident.line, ident.column,
                 ident.column + len(ident.name), ident.symbol, region, context)
        region.uses.append(u)
        return u

    def _mk_builtin_use(self, name, node, region):
        symbol = region.scope.lookup(name)
        u = _Use(self._uid(), name, node.line, node.column,
                 node.column + len(name), symbol, region, "call")
        region.uses.append(u)
        return u

    # ------------------------------------------------------------------
    # 第二阶段：控制流模拟（环境 = symbol_id -> frozenset(def_id)）
    # ------------------------------------------------------------------
    @staticmethod
    def _env_copy(env):
        return {k: set(v) for k, v in env.items()}

    @staticmethod
    def _env_merge(a, b):
        out = DataFlowAnalyzer._env_copy(a)
        for k, v in b.items():
            if k in out:
                out[k] |= v
            else:
                out[k] = set(v)
        return out

    @staticmethod
    def _env_freeze(env):
        return {k: frozenset(v) for k, v in env.items()}

    @staticmethod
    def _env_same(a, b):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(a[k] == b[k] for k in a)

    def _entry_env(self, region):
        """函数/主区域入口环境。"""
        env = {}
        # 内置函数伪定义点
        builtins = sorted(
            semantic_mod.BUILTIN_SIGNATURES.keys(),
            key=lambda n: (0 if n == "print" else 1, n),
        )
        for name in builtins:
            d = _Def(self._did(), KIND_BUILTIN, name, 0, 0, 0, None,
                     sym.SCOPE_GLOBAL, "builtin", region)
            region.defs.append(d)
        # 函数名（main 区域：全部函数；函数区域：全局函数名仍可见）
        for d in self.regions[0].nested_func_defs:
            if d.symbol is not None:
                env.setdefault(id(d.symbol), set()).add(d.id)
        # 形参入口
        for d in region.entry_params:
            if d.symbol is not None:
                env[id(d.symbol)] = {d.id}
        # 保守补充：本区域可能引用到的"外层"变量定义（跨函数的全局变量、
        # 嵌套函数捕获的外层局部）。过程内分析无法确定调用点，把外层
        # 同名定义全部作为入口可达定义，保证不漏边（可能多报，不漏报）。
        local_symbols = {id(d.symbol) for d in region.defs if d.symbol is not None}
        used_symbols = {id(u.symbol) for u in region.uses if u.symbol is not None}
        for sid in used_symbols - local_symbols:
            env.setdefault(sid, set())
            for d in self._symbol_defs.get(sid, []):
                env[sid].add(d.id)
        return env

    def _analyze_region(self, region):
        env0 = self._entry_env(region)
        if region.is_main:
            self._walk_stmts(region.node.declarations, region, dict(env0), record=True)
        else:
            self._walk_block(region.node.body, region, dict(env0), record=True)

    # -- 记录使用点到达定义的小工具 --
    def _record_use(self, u, env):
        if u.symbol is None:
            u.reaching = []
            u.status = "undefined"
            return
        if u.symbol.kind == sym.KIND_BUILTIN:
            reaching = set()
            for d in self.regions[0].defs + u.region.defs:
                if d.kind == KIND_BUILTIN and d.name == u.name:
                    reaching.add(d.id)
            u.reaching = list(reaching)
        else:
            u.reaching = list(env.get(id(u.symbol), set()))
        if not u.reaching:
            u.status = "uninit"
        elif len(u.reaching) > 1:
            u.status = "merged"
        else:
            u.status = "ok"

    def _bind_def(self, env, d):
        if d.symbol is not None:
            env[id(d.symbol)] = {d.id}

    # -- 语句序列（返回 fall-through 环境；遇 return 等不可达时返回 None） --
    def _walk_stmts(self, stmts, region, env, record, stops=None):
        cur = env
        for s in stmts:
            cur = self._walk_stmt(s, region, cur, record, stops)
            if cur is None:
                break
        return cur

    def _walk_block(self, block, region, env, record, stops=None):
        return self._walk_stmts(block.statements, region, env, record, stops)

    def _walk_stmt(self, s, region, env, record, stops):
        if env is None or s is None:
            return env
        if isinstance(s, ast.Block):
            return self._walk_block(s, region, env, record, stops)
        if isinstance(s, ast.VarDecl):
            if s.initializer:
                self._walk_expr(s.initializer, region, env, record)
            col = getattr(s, "name_column", s.column)
            decl_defs = [d for d in region.defs
                         if d.kind == KIND_DECL and d.line == s.line
                         and d.column == col and d.name == s.name]
            for d in decl_defs:
                self._bind_def(env, d)
            return env
        if isinstance(s, ast.AssignStmt):
            self._walk_assign(s, region, env, record)
            return env
        if isinstance(s, ast.ExprStmt):
            self._walk_expr(s.expr, region, env, record)
            return env
        if isinstance(s, ast.PrintStmt):
            self._record_builtin_use("print", s, region, env, record)
            for a in s.args:
                self._walk_expr(a, region, env, record)
            return env
        if isinstance(s, ast.IfStmt):
            return self._walk_if(s, region, env, record, stops)
        if isinstance(s, ast.WhileStmt):
            return self._walk_while(s, region, env, record, stops)
        if isinstance(s, ast.ForStmt):
            return self._walk_for(s, region, env, record, stops)
        if isinstance(s, ast.ReturnStmt):
            if s.value:
                self._walk_expr(s.value, region, env, record)
            if stops is not None:
                stops.setdefault("return", []).append(self._env_copy(env))
            return None
        if isinstance(s, (ast.BreakStmt, ast.ContinueStmt)):
            if stops is not None:
                key = "break" if isinstance(s, ast.BreakStmt) else "continue"
                stops.setdefault(key, []).append(self._env_copy(env))
            return None
        if isinstance(s, ast.FunctionDecl):
            # 嵌套函数声明：函数名提升在区域入口已处理；此处无需改变环境
            return env
        return env

    def _walk_if(self, s, region, env, record, stops):
        # if-elif-else 的判定条件顺序求值（elif 条件可能含赋值/副作用）
        after_cond = env
        exits = []
        for cond, body in s.branches:
            self._walk_expr(cond, region, after_cond, record)
            cond_false = self._env_copy(after_cond)
            body_out = self._walk_block(body, region, self._env_copy(after_cond),
                                        record, stops)
            if body_out is not None:
                exits.append(body_out)
            after_cond = cond_false  # 走到下一个 elif 判定的路径
        if s.else_block:
            else_out = self._walk_block(s.else_block, region, self._env_copy(after_cond),
                                        record, stops)
            if else_out is not None:
                exits.append(else_out)
        else:
            # 没有 else：所有分支条件皆假的路径直接落到汇合点
            exits.append(after_cond)
        if not exits:
            return None
        merged = exits[0]
        for e in exits[1:]:
            merged = self._env_merge(merged, e)
        return merged

    def _walk_while(self, s, region, env, record, stops):
        # 不动点：head 合并入口环境与回边（body fall-through + continue）
        head = self._env_copy(env)
        loop_stops = {}
        while True:
            frozen = self._env_freeze(head)
            self._walk_expr(s.condition, region, head, record=False)
            body_out = self._walk_block(s.body, region, self._env_copy(head),
                                        record=False, stops=loop_stops)
            back = self._env_copy(head)  # 零次迭代时条件也用到入口环境
            if body_out is not None:
                back = self._env_merge(back, body_out)
            for e in loop_stops.get("continue", []):
                back = self._env_merge(back, e)
            new_head = self._env_merge(self._env_copy(env), back)
            if self._env_same(frozen, new_head):
                head = new_head
                break
            head = new_head
            loop_stops = {}
        # 收敛后做一次"带记录"的遍历（此时循环体定义已在 head 中，
        # 回边使用点能记录到循环体内的再定义）
        final_stops = {}
        self._walk_expr(s.condition, region, head, record=record)
        body_out = self._walk_block(s.body, region, self._env_copy(head),
                                    record=record, stops=final_stops)
        # 出口：条件为假 + break
        exit_env = self._env_copy(head)
        for e in final_stops.get("break", []):
            exit_env = self._env_merge(exit_env, e)
        return exit_env

    def _walk_for(self, s, region, env, record, stops):
        cur = env
        # init 在循环之前执行一次（def 绑定幂等，直接以最终 record 模式走一遍）
        if s.init:
            cur = self._walk_stmt(s.init, region, cur, record=record, stops=stops)
            if cur is None:
                return None
        head = self._env_copy(cur)
        loop_stops = {}
        while True:
            frozen = self._env_freeze(head)
            if s.condition:
                self._walk_expr(s.condition, region, head, record=False)
            body_out = self._walk_block(s.body, region, self._env_copy(head),
                                        record=False, stops=loop_stops)
            after = body_out if body_out is not None else self._env_copy(head)
            conts = loop_stops.get("continue", [])
            for e in conts:
                after = self._env_merge(after, e)
            if s.increment:
                if isinstance(s.increment, ast.AssignStmt):
                    self._walk_assign(s.increment, region, after, record=False)
                else:
                    self._walk_expr(s.increment, region, after, record=False)
            new_head = self._env_merge(self._env_copy(cur), after)
            if self._env_same(frozen, new_head):
                head = new_head
                break
            head = new_head
            loop_stops = {}
        final_stops = {}
        if s.condition:
            self._walk_expr(s.condition, region, head, record=record)
        body_out = self._walk_block(s.body, region, self._env_copy(head),
                                    record=record, stops=final_stops)
        after = body_out if body_out is not None else self._env_copy(head)
        for e in final_stops.get("continue", []):
            after = self._env_merge(after, e)
        if s.increment:
            if isinstance(s.increment, ast.AssignStmt):
                self._walk_assign(s.increment, region, after, record=record)
            else:
                self._walk_expr(s.increment, region, after, record=record)
        exit_env = self._env_copy(head)
        if not s.condition:
            # 无界循环：正常路径不可达出口，只有 break
            exit_env = None
        if exit_env is not None:
            for e in final_stops.get("break", []):
                exit_env = self._env_merge(exit_env, e)
        else:
            for e in final_stops.get("break", []):
                exit_env = e if exit_env is None else self._env_merge(exit_env, e)
        return exit_env

    # -- 表达式（左到右，短路两侧的使用都记录） --
    def _walk_expr(self, e, region, env, record):
        if e is None:
            return
        if isinstance(e, ast.Identifier):
            u = self._find_use(region, e.line, e.column, e.name)
            if u is not None and record:
                self._record_use(u, env)
            return
        if isinstance(e, ast.AssignStmt):
            self._walk_assign(e, region, env, record)
            return
        if isinstance(e, (ast.BinaryExpr, ast.LogicalExpr)):
            self._walk_expr(e.left, region, env, record)
            self._walk_expr(e.right, region, env, record)
            return
        if isinstance(e, ast.UnaryExpr):
            self._walk_expr(e.operand, region, env, record)
            return
        if isinstance(e, ast.CallExpr):
            if isinstance(e.callee, ast.Identifier):
                u = self._find_use(region, e.callee.line, e.callee.column, e.callee.name)
                if u is not None and record:
                    self._record_use(u, env)
            else:
                self._walk_expr(e.callee, region, env, record)
            for a in e.args:
                self._walk_expr(a, region, env, record)
            return
        if isinstance(e, ast.IndexExpr):
            self._walk_expr(e.target, region, env, record)
            self._walk_expr(e.index, region, env, record)
            return
        if isinstance(e, ast.ListLiteral):
            for x in e.elements:
                self._walk_expr(x, region, env, record)

    def _walk_assign(self, s, region, env, record):
        # 复合赋值：先读左值
        if s.op != "=" and isinstance(s.target, ast.Identifier):
            u = self._find_use(region, s.target.line, s.target.column, s.target.name,
                               prefer=KIND_COMPOUND)
            if u is not None and record:
                self._record_use(u, env)
        if isinstance(s.target, ast.IndexExpr):
            self._walk_expr(s.target.target, region, env, record)
            self._walk_expr(s.target.index, region, env, record)
        self._walk_expr(s.value, region, env, record)
        # 绑定定义点（位置匹配第一阶段收集的记录）
        if isinstance(s.target, ast.Identifier):
            for d in region.defs:
                if d.line == s.target.line and d.column == s.target.column \
                        and d.name == s.target.name and d.kind in (KIND_ASSIGN,
                                                                   KIND_COMPOUND,
                                                                   KIND_UNDEFINED):
                    self._bind_def(env, d)

    def _find_use(self, region, line, column, name, prefer=None):
        """根据源码位置找回第一阶段创建的使用记录（同一位置至多一条）。"""
        for u in region.uses:
            if u.line == line and u.column == column and u.name == name:
                if prefer is None or u.context == prefer:
                    return u
        return None

    def _record_builtin_use(self, name, node, region, env, record):
        u = self._find_use(region, node.line, node.column, name, prefer="call")
        if u is not None and record:
            self._record_use(u, env)

    # ------------------------------------------------------------------
    # 驱动
    # ------------------------------------------------------------------
    def analyze(self, program, table):
        self._collect(program, table)
        def_by_id = {}
        for r in self.regions:
            for d in r.defs:
                def_by_id[d.id] = d
        for r in self.regions:
            self._analyze_region(r)
        return self.to_dict(def_by_id)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self, def_by_id=None):
        def_by_id = def_by_id or {}
        regions_out = []
        total_defs = total_uses = total_flows = 0
        for r in self.regions:
            defs = [self._def_dict(d) for d in r.defs]
            # 统计口径：定义点只算源码中真实出现的（不含内置函数伪定义）
            real_def_count = sum(1 for d in r.defs if d.line != 0)
            uses = []
            flows = []
            for u in r.uses:
                ud = self._use_dict(u)
                uses.append(ud)
                for did in u.reaching:
                    d = def_by_id.get(did)
                    # 回边：定义在源码顺序上位于使用点之后或同位。同位的
                    # 情形只会在循环中出现（如 s += i 的写入经下一轮回流到
                    # 自己的读取、for 头增量回流到同行的条件）；直线代码里
                    # 复合赋值的定义不可能到达它之前的读取，故不会误入。
                    is_back = bool(d and (
                        d.line > u.line or
                        (d.line == u.line and d.column >= u.column)))
                    flows.append({
                        "id": f"{did}->{u.id}",
                        "def": did, "use": u.id,
                        "back_edge": is_back,
                        "kind": self._flow_kind(d, u) if d else "variable",
                    })
                    total_flows += 1
            regions_out.append({
                "key": r.key,
                "name": r.name,
                "is_main": r.is_main,
                "defs": defs,
                "uses": uses,
                "flows": flows,
            })
            total_defs += real_def_count
            total_uses += len(uses)
        return {
            "regions": regions_out,
            "stats": {
                "regions": len(regions_out),
                "defs": total_defs,
                "uses": total_uses,
                "flows": total_flows,
            },
        }

    @staticmethod
    def _flow_kind(d, u):
        if d.kind == KIND_BUILTIN:
            return "builtin"
        if d.kind == KIND_FUNCTION:
            return "function"
        if d.kind == KIND_PARAM:
            return "parameter"
        return "variable"

    def _def_dict(self, d):
        return {
            "id": d.id,
            "kind": d.kind,
            "name": d.name,
            "line": d.line,
            "column": d.column,
            "end_column": d.end_column,
            "scope_type": d.scope_type,
            "scope_name": d.scope_name,
            "region": d.region.key,
            "is_const": d.is_const,
            "synthetic": d.line == 0,
        }

    def _use_dict(self, u):
        return {
            "id": u.id,
            "name": u.name,
            "line": u.line,
            "column": u.column,
            "end_column": u.end_column,
            "context": u.context,
            "status": u.status,
            "resolved": u.symbol.kind if u.symbol is not None else None,
            "reaching": list(u.reaching),
        }


# ---------------------------------------------------------------------------
# 便捷入口：源码 -> 数据流视图（复用词法/语法/语义分析）
# ---------------------------------------------------------------------------
def analyze_source(source: str):
    """返回 (view, diagnostics)。词法/语法失败时 view 为 None。"""
    diagnostics = diag.DiagnosticBag()
    tokens, lex_diags = lexer_mod.tokenize(source)
    diagnostics.items.extend(lex_diags.items)
    if lex_diags.has_errors:
        return None, diagnostics

    parser = parser_mod.Parser(tokens, diagnostics)
    program = parser.parse()
    if diagnostics.has_errors:
        return None, diagnostics

    analyzer = semantic_mod.SemanticAnalyzer()
    analyzer.set_source(source)
    analyzer.analyze(program)

    dfa = DataFlowAnalyzer(source)
    view = dfa.analyze(program, analyzer.symbols)
    view["diagnostics"] = analyzer.diagnostics.to_list()
    view["has_errors"] = analyzer.diagnostics.has_errors
    view["success"] = True  # 分析本身完成（即使存在语义诊断也产出结果）
    return view, diagnostics
