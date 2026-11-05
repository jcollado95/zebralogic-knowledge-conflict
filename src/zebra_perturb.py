"""ZebraLogic (grid_mode) template extraction + perturbation generator.

Pipeline
--------
1. parse_puzzle(text, solution)  -> Template (logical form: attributes, value domains,
   clues as typed relations over (attr, value) slots, per-slot surface phrase templates).
2. Variant(tpl)                  -> composable perturbation state (value relabelling,
   render mode, clue list/order, expected solvability).
3. Perturbation operators (all seeded, all return a new Variant):
     relabel_names_gender / gender_congruence / prior_congruence (nationality, celebrity),
     cultural_names, symbolic, madeup, implausible, shuffle_clues / reorder_clues,
     underdetermine (drop a clue -> >=2 solutions), contradict (alter a clue -> 0 solutions).
4. variant.render() -> (puzzle_text, solution{header, rows} | None)
5. validate_text(text, header_keys, expected_rows, expected) re-parses the *rendered text*
   from scratch and solves it with the CSP, so every emitted puzzle is checked end-to-end.
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
    if re.match(r'^[A-Z]\d', w):            # symbolic ids read as letters: "an A1", "an F2", "a B3"
        return w[0] in 'AEFHILMNORSX'
    return w[:1].lower() in 'aeiou'


@dataclass
class Variant:
    tpl: Template
    relabel: dict = field(default_factory=dict)     # attr -> {old_value: new_value}
    mode: str = 'surface'                            # 'surface' | 'symbolic_full'
    clues: list = None                               # clue dicts (subset / reordered / altered)
    expected: str = 'unique'                         # 'unique' | 'underdetermined' | 'contradictory'
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
        if self.mode == 'symbolic_full':
            return nv if e['attr'] == 'Name' else 'the person with ' + nv
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
            if self.mode == 'symbolic_full':
                label = 'Each person has a unique identifier' if a['key'] == 'Name' else f'Each person has a unique value of property {chr(64 + j)}'
            lines.append(f" - {label}: " + ', '.join(f"`{self.val(a['key'], v)}`" for v in a['values']))
        lines.append('\n## Clues:')
        lines += [f"{i}. {render_clue(c, self._ent)}" for i, c in enumerate(self.clues, 1)]
        text = '\n'.join(lines) + '\n'
        sol = None
        if self.expected == 'unique':
            sol = dict(header=['House'] + self.keys(),
                       rows=[[r[0]] + [self.val(a['key'], r[j + 1]) for j, a in enumerate(t.attrs)] for r in t.gold])
        return text, sol


# ----------------------------------------------------------------------------- KG / lexicons
class KG:
    def __init__(self, path):
        d = json.load(open(path, encoding='utf-8'))
        self.triples = [tuple(t) for t in d['triples']]; self.nouns = d['implausible_nouns']

    def subj(self, rel, obj): return [s for s, r, o in self.triples if r == rel and o == obj]

    def obj(self, subj, rel): return next((o for s, r, o in self.triples if s == subj and r == rel), None)


def _taken(tpl, except_attr):
    return {v.lower() for a in tpl.attrs if a['key'] != except_attr for v in a['values']}


def _pick(pool, k, rng, avoid):
    pool = [p for p in pool if p.lower() not in avoid]
    if len(pool) < k: raise ValueError('pool too small')
    return rng.sample(pool, k)


def _set_names(v, new_by_house, tag, **meta):
    """new_by_house: {house: new_name}; Name relabel follows the gold solution."""
    g = v.tpl.gold_pos()
    old_by_house = {g[('Name', x)]: x for x in v.tpl.attrs[v.tpl.keys.index('Name')]['values']}
    w = v.fork(tag); w.relabel['Name'] = {old_by_house[h]: nm for h, nm in new_by_house.items()}; w.meta.update(meta)
    return w


GENDER_ATTRS = ['Occupation', 'Hobby', 'FavoriteSport']


def gender_all(v, kg, gender, seed=0):
    rng = random.Random(seed); n = v.tpl.n
    names = _pick(kg.subj('has_gender', gender), n, rng, _taken(v.tpl, 'Name'))
    return _set_names(v, dict(zip(range(1, n + 1), names)), f'gender_all_{gender}')


def gender_congruence(v, kg, mode, seed=0):
    """Assign female/male names so that, in the gold solution, each person's gender matches ('aligned')
    or contradicts ('conflict') the stereotype of their value in the first gender-coded attribute present.
    Neutral values get a seeded gender shared by the aligned/conflict pair (matched design)."""
    attr = next((a for a in GENDER_ATTRS if a in v.tpl.keys), None)
    if attr is None: return None
    rng = random.Random(seed); g = v.tpl.gold_pos()
    flip = {'male': 'female', 'female': 'male'}
    plan = {}
    for val in v.tpl.attrs[v.tpl.keys.index(attr)]['values']:
        st = kg.obj(val, 'gender_stereotype')
        base = st or rng.choice(['male', 'female'])
        plan[g[(attr, val)]] = (base if (mode == 'aligned' or st is None) else flip[base], st, val)
    avoid = _taken(v.tpl, 'Name'); out = {}
    pools = {gd: _pick(kg.subj('has_gender', gd), v.tpl.n, random.Random(seed + 1), avoid) for gd in ('male', 'female')}
    for h in sorted(plan): out[h] = pools[plan[h][0]].pop()
    coded = sum(1 for p in plan.values() if p[1])
    return _set_names(v, out, f'gender_{mode}', bias_attr=attr, n_coded=coded,
                      assignment={str(h): dict(name=out[h], gender=p[0], value=p[2], stereotype=p[1]) for h, p in plan.items()})


PRIOR = {'Nationality': 'name_typical_of', 'FavoriteSport': 'known_for'}


def prior_congruence(v, kg, attr, mode, seed=0):
    """Names whose world-knowledge prior points to a value of `attr` (e.g. Astrid->swede, Messi->soccer).
    aligned: each person's name matches their gold value; conflict: a derangement (no person matches)."""
    if attr not in v.tpl.keys: return None
    rng = random.Random(seed); g = v.tpl.gold_pos(); rel = PRIOR[attr]
    values = v.tpl.attrs[v.tpl.keys.index(attr)]['values']
    avoid = _taken(v.tpl, 'Name')
    name_for = {val: _pick(kg.subj(rel, val), 1, rng, avoid)[0] for val in values}
    target = list(values)
    if mode == 'conflict':
        while True:
            perm = values[:]; rng.shuffle(perm)
            if all(a != b for a, b in zip(values, perm)): target = perm; break
    out = {g[(attr, val)]: name_for[t] for val, t in zip(values, target)}
    return _set_names(v, out, f'{attr.lower()}_{mode}', prior_attr=attr,
                      assignment={str(g[(attr, val)]): dict(name=name_for[t], value=val, prior=t) for val, t in zip(values, target)})


def cultural_names(v, kg, group, seed=0):
    rng = random.Random(seed); n = v.tpl.n
    names = _pick(kg.subj('cultural_group', group), n, rng, _taken(v.tpl, 'Name'))
    return _set_names(v, dict(zip(range(1, n + 1), names)), f'cultural_{group}')


def symbolic(v, full=False):
    w = v.fork('symbolic_full' if full else 'symbolic_values', mode='symbolic_full' if full else v.mode)
    for j, a in enumerate(v.tpl.attrs):
        sym = [f'P{i + 1}' for i in range(len(a['values']))] if a['key'] == 'Name' else \
              [f'{chr(64 + j)}{i + 1}' for i in range(len(a['values']))]
        w.relabel[a['key']] = dict(zip(a['values'], sym))
    return w


def _pseudoword(rng, cap):
    C, V = 'bdfgklmnprstvz', 'aeiou'
    w = ''.join(rng.choice(C) + rng.choice(V) for _ in range(rng.choice([2, 3]))) + rng.choice(['', 'n', 'r', 'l'])
    return w.capitalize() if cap else w


def madeup(v, seed=0, attrs=None):
    rng = random.Random(seed); w = v.fork('madeup'); used = {x.lower() for a in v.tpl.attrs for x in a['values']}
    for a in v.tpl.attrs:
        if attrs and a['key'] not in attrs: continue
        new = []
        while len(new) < len(a['values']):
            p = _pseudoword(rng, cap=a['key'] in ('Name', 'Mother', 'Children'))
            if p.lower() not in used and not any(p.lower() in u or u in p.lower() for u in used):
                used.add(p.lower()); new.append(p)
        w.relabel[a['key']] = dict(zip(a['values'], new))
    return w


def implausible(v, kg, seed=0, source='nouns', corpus_values=None):
    """Non-Name attributes get world-implausible fillers: inanimate nouns ('favorite food is sand') or values
    of a different attribute not present in the puzzle ('loves Dunhill books')."""
    rng = random.Random(seed); w = v.fork(f'implausible_{source}'); avoid = {x.lower() for a in v.tpl.attrs for x in a['values']}
    nouns = [x for x in kg.nouns if x.lower() not in avoid]; rng.shuffle(nouns)
    donors = [k for k in (corpus_values or {}) if k not in v.tpl.keys and k not in ('Name', 'Mother', 'Children')]
    rng.shuffle(donors)
    for a in v.tpl.attrs:
        if a['key'] == 'Name': continue
        k = len(a['values'])
        if source == 'nouns':
            new = [nouns.pop() for _ in range(k)]
        else:
            while True:  # skip donors sharing too many strings with the puzzle (e.g. Pet vs Animal)
                d = donors.pop(); pool = sorted(x for x in corpus_values[d] if x.lower() not in avoid)
                if len(pool) >= k: break
            new = rng.sample(pool, k)
            w.meta.setdefault('donors', {})[a['key']] = d
        avoid |= {x.lower() for x in new}
        w.relabel[a['key']] = dict(zip(a['values'], new))
    return w


def shuffle_clues(v, seed=0):
    rng = random.Random(seed); c = list(v.clues); rng.shuffle(c)
    return v.fork(f'shuffle_s{seed}', clues=c)


def reorder_clues(v, how):
    c = list(v.clues)
    if how == 'reverse': c = c[::-1]
    elif how == 'unary_last': c = [x for x in c if x['rel'] not in UNARY] + [x for x in c if x['rel'] in UNARY]
    elif how == 'unary_first': c = [x for x in c if x['rel'] in UNARY] + [x for x in c if x['rel'] not in UNARY]
    return v.fork(f'order_{how}', clues=c)


def underdetermine(v, seed=0, max_count=50):
    """Drop one clue such that the puzzle has >= 2 solutions."""
    rng = random.Random(seed); idx = list(range(len(v.clues))); rng.shuffle(idx); t = v.tpl
    for i in idx:
        cl = v.clues[:i] + v.clues[i + 1:]
        s = solve(t.n, t.attrs, cl, limit=max_count)
        if len(s) >= 2:
            return v.fork('underdetermined', clues=cl, expected='underdetermined',
                          meta={**v.meta, 'dropped_clue': render_clue(v.clues[i], v._ent), 'n_solutions': len(s),
                                'n_solutions_capped': len(s) >= max_count})
    return None


def _alterations(c, t, rng):
    g = t.gold_pos(); out = []
    if c['rel'] == 'found_at':
        out += [dict(c, pos=p) for p in range(1, t.n + 1) if p != c['pos']]
    elif c['rel'] == 'not_at':
        out.append(dict(c, pos=g[(c['ents'][0]['attr'], c['ents'][0]['value'])]))
    elif c['rel'] == 'same':
        # swap the 2nd argument for another value of the same attribute, reusing that value's own phrase
        # from elsewhere in the puzzle (phrases can be value-specific, e.g. 'aquarium of fish')
        e = c['ents'][1]
        seen = {(x['attr'], x['value']): x for cc in t.clues for x in cc['ents']}
        out += [dict(c, ents=[c['ents'][0], seen[(e['attr'], x)]]) for x in t.attrs[t.keys.index(e['attr'])]['values']
                if x != e['value'] and (e['attr'], x) in seen]
    else:
        flip = {'left_of': 'right_of', 'right_of': 'left_of', 'left_adj': 'right_adj', 'right_adj': 'left_adj'}
        if c['rel'] in flip: out.append(dict(c, rel=flip[c['rel']]))
        if c['rel'] in ('next_to', 'one_between', 'two_between'):
            out += [dict(c, rel=r) for r in ('next_to', 'one_between', 'two_between') if r != c['rel']]
    rng.shuffle(out); return out


def contradict(v, seed=0):
    """Alter one clue so that the clue set has no solution."""
    rng = random.Random(seed); t = v.tpl; idx = list(range(len(v.clues))); rng.shuffle(idx)
    for i in idx:
        for alt in _alterations(v.clues[i], t, rng):
            cl = v.clues[:i] + [alt] + v.clues[i + 1:]
            if not solve(t.n, t.attrs, cl, limit=1):
                return v.fork('contradictory', clues=cl, expected='contradictory',
                              meta={**v.meta, 'altered_clue_index': i + 1, 'original_clue': render_clue(v.clues[i], v._ent),
                                    'altered_clue': render_clue(alt, v._ent)})
    return None


# ----------------------------------------------------------------------------- Wikidata-backed operators
# `wkg` is a wikidata_kg.WikidataKG. All draws respect the uniqueness rules documented there.
REL_FOR_ATTR = {'Occupation': 'occupation', 'Nationality': 'citizenship', 'FavoriteSport': 'sport', 'MusicGenre': 'genre'}


def _taken_labels(v, skip):
    return {v.val(a['key'], x).lower() for a in v.tpl.attrs if a['key'] not in skip for x in a['values']}


def _name_relabel_by_house(v, names_by_house):
    g = v.tpl.gold_pos(); vals = v.tpl.attrs[v.tpl.keys.index('Name')]['values']
    return {x: names_by_house[g[('Name', x)]] for x in vals}


def _rng(v, seed, salt=''):
    """Per-puzzle generator: draws differ across puzzles but are reproducible; the salt excludes the
    aligned/conflict mode so matched pairs share their draws."""
    import hashlib
    return random.Random(int(hashlib.sha1(f'{v.tpl.id}|{seed}|{salt}'.encode()).hexdigest()[:12], 16))


def _pool_ok(wkg, k, include_low):
    spec = wkg.catalog['value_pools'].get(k, {})
    if spec.get('type') == 'same_as': spec = wkg.catalog['value_pools'].get(spec['of'], {})
    return bool(wkg.pools.get(k)) and (include_low or spec.get('quality') != 'low')


def wd_relabel(v, wkg, seed=0, attrs=None, top=None, include_low=False):
    """Fresh values for every attribute that has a Wikidata pool (names: mixed-gender given names;
    Mother: female names; Children: any). Same puzzle logic, new surface entities. `top` restricts to the
    `top` most-linked candidates (more typical values)."""
    rng = _rng(v, seed, 'relabel'); w = v.fork('wd_relabel' + (f'_top{top}' if top else '') + f'_s{seed}')
    taken = set(); n = v.tpl.n; skipped = []
    for a in v.tpl.attrs:
        k = a['key']
        if attrs and k not in attrs: taken |= {x.lower() for x in a['values']}; continue
        try:
            if k == 'Name':
                nf = rng.randint(1, n - 1)
                new = wkg.given_names('female', nf, rng, taken, top=top) + wkg.given_names('male', n - nf, rng, taken, top=top)
                rng.shuffle(new)
            elif k == 'Mother': new = wkg.given_names('female', n, rng, taken, top=top)
            elif k == 'Children': new = wkg.given_names(rng.choice(['female', 'male']), n, rng, taken, top=top)
            elif _pool_ok(wkg, k, include_low): new = wkg.values(k, n, rng, taken, top=top)
            else: raise ValueError('no pool')
        except ValueError:
            skipped.append(k); taken |= {x.lower() for x in a['values']}; continue
        w.relabel[k] = dict(zip(a['values'], new)); taken |= {x.lower() for x in new}
    w.meta['kept_original'] = skipped
    return w


def wd_names(v, wkg, seed=0, gender=None, lang=None):
    """Replace names only: all-female / all-male / a language group (P407 'language of work or name')."""
    rng = _rng(v, seed, f'names{gender}{lang}'); n = v.tpl.n; taken = _taken_labels(v, {'Name'})
    try:
        if gender:
            new = wkg.given_names(gender, n, rng, taken, lang=lang)
        else:  # language group, mixed gender
            nf = rng.randint(1, n - 1)
            fem = wkg.given_names('female', nf, rng, taken, lang=lang)
            new = fem + wkg.given_names('male', n - nf, rng, taken | {x.lower() for x in fem}, lang=lang)
            rng.shuffle(new)
    except ValueError:
        return None
    if len({x.lower() for x in new}) < n: return None
    w = v.fork(f'wd_names_{gender or "mixed"}' + (f'_{lang}' if lang else '') + f'_s{seed}')
    w.relabel['Name'] = dict(zip(v.tpl.attrs[0]['values'], new)); return w


def _target_attr(v, rel, rng):
    for a, r in REL_FOR_ATTR.items():
        if r == rel and a in v.tpl.keys: return a, False
    cands = [k for k in v.tpl.keys if k not in ('Name',) + tuple(REL_FOR_ATTR)]
    return rng.choice(cands), True


def wd_grounded(v, wkg, rel, mode, seed=0):
    """Name <- real people; one attribute <- their Wikidata values for `rel` (retyped if the puzzle lacks it).
    aligned: everyone holds their true value (and nobody else's); conflict: derangement (nobody holds theirs)."""
    rng = _rng(v, seed, rel + 'houses'); t = v.tpl; n = t.n
    attr, retyped = _target_attr(v, rel, _rng(v, seed, rel + 'target'))
    taken = _taken_labels(v, {'Name', attr})
    try:
        asg = wkg.people_assignment(rel, n, _rng(v, seed, rel), mode=mode, taken=taken)
    except ValueError:
        return None
    g = t.gold_pos(); houses = list(range(1, n + 1)); rng.shuffle(houses)
    w = v.fork(f'wd_{rel}_{mode}_s{seed}')
    av = t.attrs[t.keys.index(attr)]['values']; by_house_attr = {g[(attr, x)]: x for x in av}
    w.relabel['Name'] = _name_relabel_by_house(v, {h: a[0] for h, a in zip(houses, asg)})
    w.relabel[attr] = {by_house_attr[h]: a[2] for h, a in zip(houses, asg)}
    P = wkg.people[rel]
    if retyped: w.retype[attr] = dict(key=P['key'], line=P['line'], phrase=P['phrase'])
    w.meta.update(relation=rel, target_attr=attr, retyped=retyped,
                  assignment={str(h): dict(person=a[0], gender=a[1], value=a[2], true_values=a[3]) for h, a in zip(houses, asg)})
    return w


def wd_gender_occupation(v, wkg, mode, seed=0, n_skewed=None):
    """Occupations with the most female- / male-skewed holders in Wikidata (female share among well-known
    people) + given names whose gender matches ('aligned') or opposes ('conflict') that skew.
    aligned and conflict share the same occupations and name draws (matched pair)."""
    rng = _rng(v, seed, 'genderocc'); t = v.tpl; n = t.n
    share = wkg.gender_share('occupation', min_people=200)
    ranked = sorted(share.items(), key=lambda kv: kv[1]['female_share'])
    k = n_skewed or 8
    male_sk = [o for o, _ in ranked[:k]]; female_sk = [o for o, _ in ranked[-k:]]
    attr, retyped = _target_attr(v, 'occupation', _rng(v, seed, 'genderocc_target'))
    taken = _taken_labels(v, {'Name', attr})
    nf = n // 2 if rng.random() < 0.5 else n - n // 2
    occ_f = wkg._draw([dict(label=o) for o in female_sk], nf, rng, taken)
    occ_m = wkg._draw([dict(label=o) for o in male_sk], n - nf, rng, taken | {o['label'].lower() for o in occ_f})
    occ = [(o['label'], 'female') for o in occ_f] + [(o['label'], 'male') for o in occ_m]; rng.shuffle(occ)
    taken2 = taken | {o[0].lower() for o in occ}
    fn = wkg.given_names('female', n, rng, taken2); mn = wkg.given_names('male', n, rng, taken2 | {x.lower() for x in fn})
    flip = {'female': 'male', 'male': 'female'}
    g = t.gold_pos(); av = t.attrs[t.keys.index(attr)]['values']; by_house_attr = {g[(attr, x)]: x for x in av}
    names, plan = {}, {}
    for h, (o, sk) in zip(range(1, n + 1), occ):
        gd = sk if mode == 'aligned' else flip[sk]
        names[h] = (fn if gd == 'female' else mn).pop(0)
        plan[str(h)] = dict(name=names[h], gender=gd, occupation=o, skew=sk, female_share=round(share[o]['female_share'], 3))
    w = v.fork(f'wd_gender_{mode}_s{seed}')
    w.relabel['Name'] = _name_relabel_by_house(v, names)
    w.relabel[attr] = {by_house_attr[h]: occ[h - 1][0] for h in range(1, n + 1)}
    if retyped:
        P = wkg.people['occupation']; w.retype[attr] = dict(key=P['key'], line=P['line'], phrase=P['phrase'])
    w.meta.update(target_attr=attr, retyped=retyped, assignment=plan)
    return w


# ----------------------------------------------------------------------------- end-to-end validation
def validate_text(text, keys, rows=None, expected='unique'):
    """Re-parse rendered text from scratch, solve, and compare with the expected outcome."""
    n, _, attrs = _header(text, keys)
    clues = [parse_clue(c, attrs) for c in _clues(text)]
    s = solve(n, attrs, clues, limit=2)
    if expected == 'unique':
        return len(s) == 1 and sol_rows(n, attrs, s[0]) == rows
    if expected == 'underdetermined':
        return len(s) >= 2
    if expected == 'contradictory':
        return len(s) == 0
