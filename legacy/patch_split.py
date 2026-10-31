"""patch_split.py : makes copies of the DOFA training scripts whose make_split() honours SPLIT_MODE=disjoint.
The original scripts are not touched.  For every input X.py it writes X_disj.py, in which

    def make_split(...)        ->  def _make_split_random(...)     (unchanged body)
    + right after that function:   import split_modes as _sm; make_split = _sm.wrap(_make_split_random)

With SPLIT_MODE unset (or 'random') the copy behaves exactly like the original.

usage:  python patch_split.py train_dofa_pavia.py train_dofa_pavia_fixvar.py train_dofa_salinas.py train_dofa_salinas_fixvar.py
"""
import ast, sys

for src in sys.argv[1:]:
    text = open(src).read()
    tree = ast.parse(text)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_split"]
    if len(fn) != 1:
        raise SystemExit("%s: expected exactly one top-level 'def make_split', found %d - send me the script" % (src, len(fn)))
    fn = fn[0]
    args = [a.arg for a in fn.args.args]
    if len(args) < 5:
        raise SystemExit("%s: make_split has arguments %s, expected (gt_flat, class_num, train_num, val_num, rng)" % (src, args))
    lines = text.split("\n")
    i0, i1 = fn.lineno - 1, fn.end_lineno          # def line index, line after the function
    assert lines[i0].startswith("def make_split("), lines[i0]
    lines[i0] = lines[i0].replace("def make_split(", "def _make_split_random(", 1)
    lines.insert(i1, "import split_modes as _sm\nmake_split = _sm.wrap(_make_split_random)   # added by patch_split.py")
    out = src[:-3] + "_disj.py"
    open(out, "w").write("\n".join(lines))
    print("%s -> %s   (make_split arguments: %s)" % (src, out, ", ".join(args)))
