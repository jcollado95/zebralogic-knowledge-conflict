"""ZebraLogic (grid_mode) core: template extraction, CSP solver, renderer and round-trip validation.

1. parse_puzzle(text, solution) / build_corpus(records) -> Template (logical form: attributes, value domains,
   clues as typed relations over (attr, value) slots, per-slot surface phrase templates).
2. Variant(tpl) -> relabelling state (value relabelling, attribute retyping).
3. variant.render() -> (puzzle_text, solution{header, rows}).
4. validate_text(text, header_keys, expected_rows) re-parses the *rendered text* from scratch and
   solves it with the CSP, so every emitted puzzle is checked end-to-end.
"""
from __future__ import annotations
import re, json, random, itertools, collections, copy, string
from dataclasses import dataclass, field

ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6}
ORDW = {v: k for k, v in ORD.items()}
REL = [('found_at', r'^(?P<X>.+) is in the (?P<pos>\w+) house\.$'),
       ('not_at', r'^(?P<X>.+) is not in the (?P<pos>\w+) house\.$'),
       ('next_to', r'^(?P<X>.+) and (?P<Y>.+) are next to each other\.$'),
       ('left_adj', r'^(?P<X>.+) is directly left of (?P<Y>.+)\.$'),
       ('right_adj', r'^(?P<X>.+) is directly right of (?P<Y>.+)\.$'),
       ('left_of', r'^(?P<X>.+) is somewhere to the left of (?P<Y>.+)\.$'),
       ('right_of', r'^(?P<X>.+) is somewhere to the right of (?P<Y>.+)\.$'),
       ('one_between', r'^There is one house between (?P<X>.+) and (?P<Y>.+)\.$'),
       ('two_between', r'^There are two houses between (?P<X>.+) and (?P<Y>.+)\.$'),
       ('three_between', r'^There are three houses between (?P<X>.+) and (?P<Y>.+)\.$')]
REL_TMPL = {'found_at': '{X} is in the {pos} house.', 'not_at': '{X} is not in the {pos} house.',
            'next_to': '{X} and {Y} are next to each other.', 'left_adj': '{X} is directly left of {Y}.',
            'right_adj': '{X} is directly right of {Y}.', 'left_of': '{X} is somewhere to the left of {Y}.',
            'right_of': '{X} is somewhere to the right of {Y}.', 'one_between': 'There is one house between {X} and {Y}.',
            'two_between': 'There are two houses between {X} and {Y}.',
            'three_between': 'There are three houses between {X} and {Y}.', 'same': '{X} is {Y}.'}
UNARY = {'found_at', 'not_at'}
# Irregular surface forms used by the ZebraLogic generator (value -> surface strings in clues)
ALIAS = {'ford f150': ['Ford F-150'], 'roses': ['rose bouquet'], 'swede': ['Swedish'], 'brit': ['British'],
         'dane': ['Danish'], 'painting': ['paints'], 'hip hop': ['hip-hop'], 'carnations': ['carnation']}
LEARNED = collections.defaultdict(collections.Counter)   # slot template (lower) -> attr counts


# ----------------------------------------------------------------------------- parsing
def _header(text, keys):
    pre = text.split('\n - ')[0]
    hl = [l for l in text.split('\n') if l.startswith(' - ')]
    assert len(hl) == len(keys), (len(hl), keys)
    attrs = [dict(key=k, line=re.sub(r':\s*`.*$', '', l[3:]), values=re.findall(r'`([^`]*)`', l))
             for k, l in zip(keys, hl)]
    n = int(re.search(r'There are (\d+) houses', text).group(1))
    return n, pre, attrs


def _clues(text):
    body = text.split('## Clues:')[1].strip()
    return [re.sub(r'^\d+\.\s*', '', l).strip() for l in body.split('\n') if l.strip()]


def _mentions(s, attrs, prefix=False):
    hits = set()
    tail = r'\w*' if prefix else r'(?:s|es)?'
    for a in attrs:
        for v in a['values']:
            for form in [v] + ALIAS.get(v.lower(), []):
                for m in re.finditer(r'(?<!\w)' + re.escape(form) + tail + r'(?!\w)', s, flags=re.I):
                    hits.add((m.start(), m.end(), a['key'], v, m.group(0)))
    return [h for h in hits if not any(o[0] <= h[0] and o[1] >= h[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]


def _well_formed(s, surf):
    if len(re.findall(r'\bthe person\b', s, re.I)) > 1 or re.search(r' is the person\b', s, re.I):
        return False
    return s.lower() == surf.lower() or re.match(r'^the\b', s, re.I) is not None


def _slot(s, attrs):
    h = _mentions(s, attrs) or _mentions(s, attrs, prefix=True)
    if len(h) > 1:  # same string in two attributes (e.g. fish: Pet/Animal) -> use learned phrase->attr
        h2 = [x for x in h if LEARNED.get((s[:x[0]] + '{v}' + s[x[1]:]).lower(), {}).get(x[2])]
        if len(h2) == 1:
            h = h2
    if len(h) != 1:
        return None
    st, en, k, v, surf = h[0]
    if not _well_formed(s, surf):
        return None
    return dict(attr=k, value=v, surface=surf, tmpl=s[:st] + '{v}' + s[en:])


def parse_clue(c, attrs):
    for rel, pat in REL:
        m = re.match(pat, c)
        if m:
            d = m.groupdict()
            ents = [_slot(d[x], attrs) for x in ('X', 'Y') if x in d]
            if all(ents):
                return dict(rel=rel, ents=ents, pos=ORD.get(d.get('pos')))
    cands = []
    for m in re.finditer(r' is ', c):
        a, b = _slot(c[:m.start()], attrs), _slot(c[m.end():].rstrip('.'), attrs)
        if a and b:
            cands.append(dict(rel='same', ents=[a, b], pos=None))
    if len(cands) == 1:
        return cands[0]
    raise ValueError(('ambiguous' if cands else 'unparsed') + ': ' + c)


@dataclass
class Template:
    id: str
    n: int
    preamble: str
    attrs: list            # [{key, line, values}]
    clues: list            # [{rel, ents:[{attr,value,surface,tmpl}], pos}]
    gold: list             # rows [[house, v_attr1, ...]]

    @property
    def keys(self): return [a['key'] for a in self.attrs]

    def gold_pos(self):    # (attr, value) -> house
        return {(a['key'], r[j + 1]): int(r[0]) for r in self.gold for j, a in enumerate(self.attrs)}

    def to_json(self):
        """Variable template: values replaced by slot ids {Attr.i} (i = index in header list)."""
        idx = {(a['key'], v): f"{{{a['key']}.{i}}}" for a in self.attrs for i, v in enumerate(a['values'])}
        ent = lambda e: e['tmpl'].replace('{v}', idx[(e['attr'], e['value'])])
        return dict(id=self.id, n_houses=self.n, preamble=self.preamble,
                    attributes=[dict(key=a['key'], line=a['line'], slots=[idx[(a['key'], v)] for v in a['values']],
                                     original_values=a['values']) for a in self.attrs],
                    clues=[dict(rel=c['rel'], pos=c['pos'], text=render_clue(c, ent),
                                args=[idx[(e['attr'], e['value'])] for e in c['ents']],
                                phrase_templates=[e['tmpl'] for e in c['ents']]) for c in self.clues],
                    gold=[[r[0]] + [idx[(a['key'], r[j + 1])] for j, a in enumerate(self.attrs)] for r in self.gold])


def parse_puzzle(pid, text, solution):
    keys = list(solution['header'])[1:]
    n, pre, attrs = _header(text, keys)
    rows = [[str(x) for x in r] for r in solution['rows']]
    for j, a in enumerate(attrs):
        assert set(a['values']) == {r[j + 1] for r in rows}, (pid, a['key'])
    return Template(pid, n, pre, attrs, [parse_clue(c, attrs) for c in _clues(text)], rows)


def build_corpus(records):
    """records: iterable of (id, puzzle_text, solution). Two passes: learn phrase->attr map, then parse."""
    records = list(records)
    for pid, t, s in records:  # pass 1: unambiguous slots only
        keys = list(s['header'])[1:]; _, _, attrs = _header(t, keys)
        for c in _clues(t):
            try:
                for e in parse_clue(c, attrs)['ents']:
                    LEARNED[e['tmpl'].lower()][e['attr']] += 1
            except ValueError:
                pass
    return [parse_puzzle(pid, t, s) for pid, t, s in records]


# ----------------------------------------------------------------------------- solver
def _check(c, P):
    x = P[(c['ents'][0]['attr'], c['ents'][0]['value'])]
    if c['rel'] in UNARY:
        return (x == c['pos']) == (c['rel'] == 'found_at')
    y = P[(c['ents'][1]['attr'], c['ents'][1]['value'])]
    r = c['rel']
    return {'same': x == y, 'next_to': abs(x - y) == 1, 'left_adj': x == y - 1, 'right_adj': x == y + 1,
            'left_of': x < y, 'right_of': x > y, 'one_between': abs(x - y) == 2,
            'two_between': abs(x - y) == 3, 'three_between': abs(x - y) == 4}[r]


def solve(n, attrs, clues, limit=2):
    """Backtracking over per-attribute permutations; returns up to `limit` solutions {(attr,val): house}."""
    keys = [a['key'] for a in attrs]; vals = {a['key']: a['values'] for a in attrs}
    need = [frozenset(e['attr'] for e in c['ents']) for c in clues]
    perms = {}
    for k in keys:  # pre-filter with clues internal to one attribute
        own = [c for c, nd in zip(clues, need) if nd == {k}]
        ok = []
        for p in itertools.permutations(range(1, n + 1)):
            P = {(k, v): h for v, h in zip(vals[k], p)}
            if all(_check(c, P) for c in own): ok.append(P)
        perms[k] = ok
    order, left = [], set(keys)
    while left:  # most-connected-to-assigned first, then fewest candidates
        k = max(left, key=lambda k: (sum(1 for nd in need if k in nd and len(nd) > 1 and (nd - {k}) <= set(order)),
                                     -len(perms[k])))
        order.append(k); left.remove(k)
    act = [[c for c, nd in zip(clues, need) if k in nd and nd <= set(order[:i + 1]) and len(nd) > 1]
           for i, k in enumerate(order)]
    sols, P = [], {}

    def rec(i):
        if len(sols) >= limit: return
        if i == len(order): sols.append(dict(P)); return
        for cand in perms[order[i]]:
            P.update(cand)
            if all(_check(c, P) for c in act[i]): rec(i + 1)
        for kv in perms[order[i]][0] if perms[order[i]] else []: P.pop(kv, None)
    rec(0)
    return sols


def sol_rows(n, attrs, S):
    return [[str(h)] + [next(v for v in a['values'] if S[(a['key'], v)] == h) for a in attrs] for h in range(1, n + 1)]


# ----------------------------------------------------------------------------- rendering / variants
def render_clue(c, ent_fn):
    d = {'X': ent_fn(c['ents'][0])}
    if len(c['ents']) > 1: d['Y'] = ent_fn(c['ents'][1])
    if c['pos']: d['pos'] = ORDW[c['pos']]
    s = REL_TMPL[c['rel']].format(**d)
    return s[0].upper() + s[1:] if s[0].islower() and not c['ents'][0]['tmpl'].startswith('{v}') else s


def _vowel_sound(w):
    if re.match(r'^[A-Z]\d', w):            # codes read as letters: "an A1", "an F2", "a B3"
        return w[0] in 'AEFHILMNORSX'
    return w[:1].lower() in 'aeiou'


@dataclass
class Variant:
    tpl: Template
    relabel: dict = field(default_factory=dict)     # attr -> {old_value: new_value}
    clues: list = None                               # clue dicts to render (default: the template's clues)
    tags: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    retype: dict = field(default_factory=dict)       # attr -> {key, line, phrase}: attribute replaced by a new relation

    def __post_init__(self):
        if self.clues is None: self.clues = list(self.tpl.clues)

    def fork(self, tag, **kw):
        v = copy.copy(self); v.relabel = copy.deepcopy(self.relabel); v.clues = list(self.clues)
        v.tags = self.tags + [tag]; v.meta = copy.deepcopy(self.meta); v.retype = copy.deepcopy(self.retype)
        for k, x in kw.items(): setattr(v, k, x)
        return v

    def val(self, attr, v): return self.relabel.get(attr, {}).get(v, v)

    def keys(self): return [self.retype.get(k, {}).get('key', k) for k in self.tpl.keys]

    def _ent(self, e):
        nv = self.val(e['attr'], e['value'])
        if e['attr'] in self.retype:
            e = dict(e, tmpl=self.retype[e['attr']]['phrase'])
        elif nv == e['value']:
            return e['tmpl'].replace('{v}', e['surface'])
        def art(m):  # re-agree the indefinite article with the substituted value
            a = 'an' if _vowel_sound(nv) else 'a'
            return (a.capitalize() if m.group(1)[0].isupper() else a) + ' ' + nv
        return re.sub(r'\b(an?) \{v\}', art, e['tmpl'], flags=re.I).replace('{v}', nv)

    def render(self):
        t = self.tpl; lines = [t.preamble]
        for j, a in enumerate(t.attrs):
            label = self.retype.get(a['key'], {}).get('line', a['line'])
            lines.append(f" - {label}: " + ', '.join(f"`{self.val(a['key'], v)}`" for v in a['values']))
        lines.append('\n## Clues:')
        lines += [f"{i}. {render_clue(c, self._ent)}" for i, c in enumerate(self.clues, 1)]
        text = '\n'.join(lines) + '\n'
        sol = dict(header=['House'] + self.keys(),
                   rows=[[r[0]] + [self.val(a['key'], r[j + 1]) for j, a in enumerate(t.attrs)] for r in t.gold])
        return text, sol


def validate_text(text, keys, rows):
    """Re-parse rendered text from scratch, solve it, and check it has exactly one solution, equal to `rows`."""
    n, _, attrs = _header(text, keys)
    clues = [parse_clue(c, attrs) for c in _clues(text)]
    s = solve(n, attrs, clues, limit=2)
    return len(s) == 1 and sol_rows(n, attrs, s[0]) == rows
