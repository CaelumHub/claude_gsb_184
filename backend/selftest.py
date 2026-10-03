# -*- coding: utf-8 -*-
"""
自检套件：验证编译器前端、解释器、调试器、剖析器、内存模型、存储层。

``python3 backend/run.py --check`` 会运行这里的所有用例并打印结果。
每个用例独立、幂等，不依赖外部网络。
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from . import compiler
from . import vm as vm_mod
from . import debugger as debugger_mod
from . import profiler as profiler_mod
from . import storage
from . import memory_model
from . import diagnostics as diag
from . import dataflow as df_mod
from . import lexer as lexer_mod
from . import parser as parser_mod
from . import semantic as semantic_mod


_RESULTS = []


def _check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond), detail))
    return bool(cond)


def _run(source, **opts):
    return _SVC_RUN(source, opts)


_SVC_RUN = None


def run_all():
    global _SVC_RUN
    from . import service
    svc = service.Service()
    _SVC_RUN = svc.run

    _test_lexer()
    _test_parser()
    _test_semantic()
    _test_dataflow()
    _test_vm_basic()
    _test_functions_recursion()
    _test_control_flow()
    _test_lists()
    _test_runtime_errors()
    _test_debugger()
    _test_profiler()
    _test_memory_model()
    _test_storage()
    _test_concurrent_writes()
    _test_full_pipeline()

    passed = 0
    for name, ok, detail in _RESULTS:
        mark = "✔" if ok else "✘"
        print(f"  {mark} {name}" + (f"  — {detail}" if detail and not ok else ""))
        if ok:
            passed += 1
    total = len(_RESULTS)
    print(f"\n自检结果：{passed}/{total} 通过")
    return passed == total


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def _test_lexer():
    from . import lexer as lexer_mod
    toks, d = lexer_mod.tokenize("var x = 3.14;\nfunc f(a) { return a + 1; }")
    types = [t.type for t in toks]
    ok = (d.has_errors is False and "var" in types and "IDENT" in types
          and "NUMBER" in types and "func" in types and "EOF" in types)
    _check("词法分析：识别关键字/标识符/数字/EOF", ok, str(types))


def _test_parser():
    res = compiler.compile_source("var x = 1 + 2 * 3;")
    ok = res.ast is not None and not res.diagnostics.has_errors
    _check("语法分析：递归下降构建 AST", ok)


def _test_semantic():
    res = compiler.compile_source("var a = 1;\nprint(b);")
    errs = res.diagnostics.errors()
    ok = len(errs) >= 1 and any("未定义" in e.message and "b" in e.message for e in errs)
    _check("语义分析：未定义变量报错 + did-you-mean", ok,
           str([e.message for e in errs]) if not ok else "")

    res2 = compiler.compile_source("func f(a, b) { return a + b; }\nf(1);")
    errs2 = res2.diagnostics.errors()
    ok2 = any("2 个参数" in e.message or "需要 2" in e.message for e in errs2)
    _check("语义分析：参数个数不匹配报错", ok2,
           str([e.message for e in errs2]) if not ok2 else "")


def _dataflow(src):
    toks, _ = lexer_mod.tokenize(src)
    prog = parser_mod.Parser(toks).parse()
    an = semantic_mod.SemanticAnalyzer()
    an.set_source(src)
    an.analyze(prog)
    return df_mod.analyze(prog, an.symbols, toks)


def _test_dataflow():
    # 1) 顺序多次赋值：后一个使用只看到最近一次再定义
    res = _dataflow("var x = 1;\nx = 2;\nprint(x);\n")
    u = next(u for u in res["uses"] if u["context"] != "callee")
    ok = u["reaching"] and all(
        (d["kind"], d["line"]) == ("reassign", 2) for d in
        [next(x for x in res["defs"] if x["id"] == i) for i in u["reaching"]])
    _check("数据流：多次赋值使用点指向最近再定义", ok, str(u["reaching"]))

    # 2) 分支汇合：if/else 两条路径的定义都到达汇合点
    src = ("var x = 1;\n"
           "if (x > 0) { x = 3; } else { x = 4; }\n"
           "print(x);\n")
    res = _dataflow(src)
    u = next(u for u in res["uses"] if u["name"] == "x" and u["line"] == 3)
    dmap = {d["id"]: d for d in res["defs"]}
    cols = sorted(dmap[i]["column"] for i in u["reaching"])
    kinds = sorted(dmap[i]["kind"] for i in u["reaching"])
    _check("数据流：if/else 分支汇合到达两个再赋值定义",
           kinds == ["reassign", "reassign"] and cols == [14, 30],
           f"{kinds} {cols}")

    # 3) 循环再赋值：循环体内使用同时到达循环前定义与循环体上一轮再定义
    src = ("var s = 0;\n"
           "while (s < 3) { s = s + 1; }\n"
           "print(s);\n")
    res = _dataflow(src)
    dmap = {d["id"]: d for d in res["defs"]}
    rhs = next(u for u in res["uses"] if u["name"] == "s" and u["line"] == 2 and u["column"] == 21)
    labels = sorted(dmap[i]["kind"] + "@L" + str(dmap[i]["line"]) for i in rhs["reaching"])
    _check("数据流：循环体使用到达循环前定义与上一轮再定义",
           labels == ["decl@L1", "reassign@L2"], str(labels))
    # 循环条件首次执行只看到循环前定义，但经回边也看到上一轮再定义
    cond = next(u for u in res["uses"] if u["name"] == "s" and u["line"] == 2 and u["column"] == 8)
    labels = sorted(dmap[i]["kind"] for i in cond["reaching"])
    _check("数据流：循环条件经回边到达再定义", labels == ["decl", "reassign"], str(labels))

    # 4) 作用域遮蔽：内层同名符号的使用只到内层定义，外层使用到外层定义
    src = ("var x = 1;\n"
           "func f() {\n"
           "    if (true) { var x = 9; print(x); }\n"
           "    print(x);\n"
           "}\n"
           "print(x);\n"
           "f();\n")
    res = _dataflow(src)
    dmap = {d["id"]: d for d in res["defs"]}
    inner = next(u for u in res["uses"] if u["region"] == "f" and u["line"] == 3)
    outer_in_f = next(u for u in res["uses"] if u["region"] == "f" and u["line"] == 4)
    global_use = next(u for u in res["uses"] if u["region"] == "<global>" and u["line"] == 6)
    reach_of = lambda u: [(dmap[i]["kind"], dmap[i]["line"]) for i in u["reaching"]]
    ok = (reach_of(inner) == [("decl", 3)]
          and reach_of(outer_in_f) == [("decl", 1)]
          and reach_of(global_use) == [("decl", 1)])
    _check("数据流：作用域遮蔽不跨符号串线", ok,
           f"{reach_of(inner)} {reach_of(outer_in_f)} {reach_of(global_use)}")

    # 5) 形参：函数体内形参使用指向形参定义
    res = _dataflow("func add(a, b) { return a + b; }\nprint(add(1, 2));\n")
    pa = next(u for u in res["uses"] if u["region"] == "add" and u["name"] == "a")
    pb = next(u for u in res["uses"] if u["region"] == "add" and u["name"] == "b")
    dmap = {d["id"]: d for d in res["defs"]}
    ok = (len(pa["reaching"]) == 1 and dmap[pa["reaching"][0]]["kind"] == "param"
          and len(pb["reaching"]) == 1 and dmap[pb["reaching"][0]]["kind"] == "param")
    _check("数据流：形参使用点指向形参定义", ok)

    # 6) 未解析名字不臆造定义；未初始化声明单独标记
    res = _dataflow("print(missing);\nvar y;\nprint(y);\n")
    miss = next(u for u in res["uses"] if u["name"] == "missing")
    y = next(u for u in res["uses"] if u["name"] == "y")
    dmap = {d["id"]: d for d in res["defs"]}
    ok = (miss["unresolved"] and miss["reaching"] == []
          and len(y["reaching"]) == 1 and dmap[y["reaching"][0]]["kind"] == "uninit")
    _check("数据流：未解析名字无定义、未初始化声明显式标记", ok)

    # 7) for 循环增量：条件处 i 到达初值定义与增量再定义（回边）
    src = "for (var i = 0; i < 3; i = i + 1) { print(i); }\n"
    res = _dataflow(src)
    cond_i = next(u for u in res["uses"] if u["line"] == 1 and u["column"] == 17)
    dmap = {d["id"]: d for d in res["defs"]}
    kinds = sorted(dmap[i]["kind"] for i in cond_i["reaching"])
    _check("数据流：for 条件到达初值声明与增量再定义", kinds == ["decl", "reassign"],
           str(kinds))


def _test_vm_basic():
    out = _run("var x = 2 + 3 * 4;\nprint(x);\nprint(\"hello\");")
    ok = out.get("ok") and out["output"] == ["14", "hello"]
    _check("解释器：算术与字符串输出", ok, str(out.get("output")))


def _test_functions_recursion():
    src = "func fib(n) { if (n < 2) { return n; } return fib(n-1) + fib(n-2); }\nprint(fib(10));"
    out = _run(src)
    ok = out.get("ok") and out["output"] == ["55"]
    _check("解释器：递归函数（fib(10)=55）", ok, str(out.get("output")))


def _test_control_flow():
    src = ("var s = 0;\n"
           "for (var i = 0; i < 5; i = i + 1) { s = s + i; }\n"
           "var w = 0; var k = 0;\n"
           "while (k < 3) { w = w + 10; k = k + 1; }\n"
           "var m = 0;\n"
           "if (s > 5) { m = 100; } elif (s > 2) { m = 50; } else { m = 1; }\n"
           "print(s, w, m);")
    out = _run(src)
    ok = out.get("ok") and out["output"] == ["10 30 100"]
    _check("解释器：for/while/if-elif-else 控制流", ok, str(out.get("output")))


def _test_lists():
    src = ("var a = [1, 2, 3];\n"
           "push(a, 4);\n"
           "a[0] = 99;\n"
           "print(len(a), a[0], a[3]);")
    out = _run(src)
    ok = out.get("ok") and out["output"] == ["4 99 4"]
    _check("解释器：列表构建/下标/len/push", ok, str(out.get("output")))


def _test_runtime_errors():
    out = _run("var x = 1 / 0;")
    ok1 = (not out.get("ok") or out.get("error") is not None) and out.get("error") is not None
    _check("运行时错误：除以零产生诊断", ok1)

    out2 = _run("var a = [1];\nprint(a[5]);")
    ok2 = out2.get("error") is not None and "越界" in out2["error"].get("message", "")
    _check("运行时错误：下标越界产生诊断", ok2)

    out3 = compiler.compile_source("var radius = 3;\nprint(radus);")
    errs3 = out3.diagnostics.errors()
    ok3 = any("radus" in e.message and e.fix for e in errs3)
    _check("错误诊断：未定义变量带 did-you-mean 修复建议", ok3,
           str([(e.message, e.fix) for e in errs3]) if not ok3 else "")


def _test_debugger():
    src = ("var total = 0;\n"
           "for (var i = 0; i < 3; i = i + 1) {\n"
           "    total = total + i;\n"
           "}\n"
           "print(total);")
    res = compiler.compile_source(src)
    assert res.success, "编译失败"
    vm = vm_mod.VM(res.bytecode, res.source_lines)
    dbg = debugger_mod.Debugger(vm, {2})
    dbg.start()
    snap = dbg.snapshot()
    ok_bp = (snap["reason"] == "breakpoint" and snap["next_instruction"] is not None
             and snap["next_instruction"]["line"] == 2)
    _check("调试器：断点在指定行暂停", ok_bp, str(snap["next_instruction"]))

    # 单步进入循环体
    dbg.step_over()
    snap2 = dbg.snapshot()
    ok_step = not vm.finished and snap2["call_stack"]
    _check("调试器：单步跳过推进到下一行", ok_step)

    # 清除断点后继续到结束（否则 for 行断点会每轮迭代重新命中）
    dbg.clear_breakpoints()
    dbg.continue_()
    snap3 = dbg.snapshot()
    ok_finish = snap3["finished"] is True and vm.output == ["3"]
    _check("调试器：继续运行到程序结束", ok_finish, str(vm.output))


def _test_profiler():
    src = ("func work() { var s = 0; for (var i = 0; i < 100; i = i + 1) { s = s + i; } return s; }\n"
           "print(work());\nprint(work());")
    out = _run(src, profile=True, sample=True, sample_interval_ms=0.2)
    prof = out.get("profile")
    ok = prof is not None and prof.get("total_instructions", 0) > 0
    fn_names = [f["name"] for f in prof.get("functions", [])]
    ok2 = any("work" in n for n in fn_names)
    _check("剖析器：插桩统计函数调用与指令数", ok and ok2, str(fn_names) if not (ok and ok2) else "")


def _test_memory_model():
    src = "var a = [1, 2, 3];\nvar b = [a, 99];\nprint(b[0]);"
    out = _run(src, memory=True)
    mem = out.get("memory")
    ok = mem is not None and mem["stats"]["object_count"] >= 2
    refs = [o for o in mem.get("objects", []) if o.get("refs")]
    _check("内存模型：堆快照包含对象与引用", ok and bool(refs), str(mem.get("stats")) if not ok else "")


def _test_storage():
    import tempfile
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "a", "b", "test.json")
    storage.write_json(path, {"x": 1, "items": [1, 2, 3]})
    data = storage.read_json(path)
    ok = data == {"x": 1, "items": [1, 2, 3]}
    # 原子写不残留临时文件
    leftovers = [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".tmp-")]
    _check("存储：JSON 原子写与读取（无临时残留）", ok and not leftovers)


def _test_concurrent_writes():
    import tempfile
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "counter.json")
    storage.write_json(path, {"count": 0})
    errors = []

    def bump(n):
        try:
            for _ in range(n):
                storage.update_json(path, lambda d: ({"count": d["count"] + 1}, True))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=bump, args=(200,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = storage.read_json(path)["count"]
    _check("存储：8 线程并发累加 1600 次无丢失", final == 1600 and not errors, f"count={final}")


def _test_full_pipeline():
    from . import service
    svc = service.Service()
    p = svc.create_project("自检项目", "var z = 6 * 7;\nprint(z);")
    v = svc.save_version(p["id"], "var z = 6 * 7;\nprint(z);", "v2")
    vers = svc.list_versions(p["id"])
    rec = svc.record_run(p["id"], v["id"], "var z = 6 * 7;\nprint(z);")
    ok = (p is not None and len(vers) >= 2 and rec.get("ok") and rec.get("output") == ["42"])
    # 调试会话
    state = svc.debug_start("var n = 1;\nprint(n);", [2])
    ok2 = state.get("ok") and state.get("reason") == "breakpoint"
    svc.delete_project(p["id"])
    _check("全链路：项目/版本/运行记录/调试会话", ok and ok2)
