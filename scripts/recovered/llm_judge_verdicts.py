"""Recovered notebook code that produced `judge_verdicts.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import json, re, os, shutil, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Setup: copy dependency files
shutil.copy('data/persons/unique_persons.csv', 'unique_persons.csv')
shutil.copy('data/puzzles/zebralogic_persons.parquet', 'zebralogic_persons.parquet')
shutil.copy('src/run_eval.py', 'run_eval.py')

import sys
sys.path.insert(0, '.')
import run_eval as E

# Build truth dict
import csv
REL = {'place of birth': 'BirthPlace', 'occupation': 'Occupation', 'mother': 'Mother',
       'educated at': 'EducatedAt', 'height (cm)': 'Height'}
truth = {r['person']: {REL[c]: r[c] for c in REL}
         for r in csv.DictReader(open('unique_persons.csv', encoding='utf-8-sig'))}

# knowledge_probe.py content (inline)
import unicodedata

URL = 'http://ada01.ujaen.es:8080/v1/chat/completions'
Q = {'BirthPlace': 'Where was {p} born? Answer with the place name only.',
     'Occupation': 'What is the main occupation of {p}? Answer with the occupation only.',
     'Mother': "What is the name of {p}'s mother? Answer with her full name only.",
     'EducatedAt': 'Which school or university did {p} attend? Answer with the institution name only.',
     'Height': 'How tall is {p} in centimetres? Answer with a number only.'}
SYN = {'association football player': ['soccer', 'footballer', 'football player'], 'film director': ['filmmaker', 'director'],
       'film actor': ['actor', 'actress'], 'actor': ['actress'], 'athletics competitor': ['runner', 'athlete', 'long-distance'],
       'businessperson': ['businessman', 'entrepreneur'], 'military officer': ['marshal', 'general', 'military'],
       'monarch': ['queen'], 'professional wrestler': ['wrestler'], 'competitive diver': ['diver'], 'artistic gymnast': ['gymnast'],
       'ice hockey player': ['hockey player'], 'Formula One driver': ['racing driver', 'formula one', 'formula 1']}
REFUSAL = re.compile(r"do not have|don't have|not able|cannot|can't|no information|not aware|unable|^none$|not publicly|i'm not sure|unknown|please provide|please specify|not available", re.I)
STOP = {'of', 'the', 'de', 'la', 'del', 'y', 'and', 'jr'}


def toks(s):
    return re.sub(r'[^a-z0-9 ]', ' ', unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()).split()


def refusal(ans):
    return bool(REFUSAL.search(ans.strip()))


def correct(rel, gold, ans):
    if refusal(ans): return False
    if rel == 'Height':
        m = re.search(r'\d{3}', ans); return bool(m) and abs(int(m.group()) - int(gold)) <= 3
    g, a = toks(gold), toks(ans)
    if not a: return False
    if rel == 'Occupation':
        return ' '.join(g) in ' '.join(a) or any(s in ans.lower() for s in SYN.get(gold, []))
    if rel == 'BirthPlace':
        return ' '.join(g) in ' '.join(a) or ' '.join(a) in ' '.join(g)
    gc = [t for t in g if t not in STOP and len(t) > 1] or g
    return all(t in a for t in gc)


def ask(model, key, p, rel, retries=3):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': Q[rel].format(p=p)}], max_tokens=60,
                           temperature=0)).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    for k in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())['choices'][0]['message']['content'] or ''
        except Exception as e:
            err = repr(e); time.sleep(10 * (k + 1))
    return 'ERROR ' + err


# Load probe data for E4B (from artifact)
kp_e4b = pd.read_csv('data/persons/knowledge_probe.csv')

# Run 31B probe
MODEL_31B = 'gemma-4-31B-it'
people = sorted(kp_e4b.person.unique())

jobs = [(p, r) for p in people for r in Q]
KEY = os.environ['VLLM_API_KEY']
with ThreadPoolExecutor(8) as ex:
    ans_31b = list(ex.map(lambda pr: ask(MODEL_31B, KEY, *pr), jobs))

r31 = pd.DataFrame([dict(model=MODEL_31B, person=p, relation=r, gold=truth[p][r], answer=a.strip()[:300],
                          refusal=refusal(a), correct=correct(r, truth[p][r], a))
                     for (p, r), a in zip(jobs, ans_31b)])

# Fix synonyms and re-score
def correct2_31b(row):
    if row.relation == 'Occupation':
        SYN_OCC = {'association football player': ['soccer', 'footballer', 'football player'],
                   'film director': ['filmmaker', 'director'], 'film actor': ['actor', 'actress'],
                   'actor': ['actress'], 'athletics competitor': ['runner', 'athlete', 'long-distance'],
                   'businessperson': ['businessman', 'entrepreneur'], 'military officer': ['marshal', 'general', 'military'],
                   'monarch': ['queen'], 'professional wrestler': ['wrestler'], 'competitive diver': ['diver'],
                   'artistic gymnast': ['gymnast'], 'ice hockey player': ['hockey player'],
                   'Formula One driver': ['racing driver', 'formula one', 'formula 1']}
        if any(s in row.answer.lower() for s in SYN_OCC.get(row.gold, [])): return True
    return row.correct

r31['correct'] = r31.apply(correct2_31b, axis=1)

# Candidates_all.csv from candidates built in session - we need the place of birth fields for the judge
# We'll build from unique_persons.csv since that's what's available
# The judge step uses todo which comes from pr (combined probe dataframe)
pr_e4b = kp_e4b[['person', 'relation', 'gold', 'answer', 'correct', 'refusal']].copy()
pr_e4b['model'] = 'gemma-4-E4B-it'

pr_31b = r31[['model', 'person', 'relation', 'gold', 'answer', 'refusal', 'correct']].copy()

# Need qid - use person as proxy since unique_persons only has person column
# Build a qid mapping from person name
people_data = list(csv.DictReader(open('unique_persons.csv', encoding='utf-8-sig')))
person_to_qid = {p['person']: i for i, p in enumerate(people_data)}

pr_e4b['qid'] = pr_e4b.person.map(person_to_qid)
pr_31b['qid'] = pr_31b.person.map(person_to_qid)

pr = pd.concat([pr_e4b, pr_31b], ignore_index=True)

# Build candidates_all info (place of birth accepted and wikidata)
# From unique_persons.csv we have 'place of birth' column directly
people_df = pd.DataFrame(people_data)
people_df['qid'] = people_df.person.map(person_to_qid)
bp_map = people_df.set_index('person')['place of birth'].to_dict()

# LLM judge for Occupation and BirthPlace non-correct non-refusal answers
JUDGE = 'gemma-4-31B-it'
JQ = {'Occupation': ("A person's occupations according to Wikidata are: {gold}.\nSomeone described this person's main occupation as: \"{ans}\".\n"
                     "Is that description correct for this person, i.e. the same occupation, a synonym, or a more general or more specific term for one of the listed occupations "
                     "(e.g. 'cosmonaut' for astronaut, 'king' for monarch, 'musician' for guitarist)? Answer YES or NO only."),
      'BirthPlace': ("A person's place of birth according to Wikidata is: {gold}.\nSomeone said this person was born in: \"{ans}\".\n"
                     "Is that answer correct, i.e. it names the same place, an alternative name for it, or the city/town/municipality that contains it? "
                     "Naming only a country, state, province or region is NOT correct, and a different city is NOT correct. Answer YES or NO only.")}


def judge(args):
    rel, gold, a = args
    body = json.dumps(dict(model=JUDGE, messages=[{'role': 'user', 'content': JQ[rel].format(gold='; '.join(gold), ans=a)}],
                           max_tokens=5, temperature=0)).encode()
    rq = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'})
    for k_ in range(3):
        try:
            return json.loads(urllib.request.urlopen(rq, timeout=300).read())['choices'][0]['message']['content'].strip().upper()
        except Exception as e:
            time.sleep(5)
    return 'ERROR'


todo = pr[pr.relation.isin(['Occupation', 'BirthPlace']) & ~pr.correct & ~pr.refusal].copy()

# Build jgold: list of accepted gold values
def build_jgold(row):
    if row.relation == 'Occupation':
        return [row.gold]
    else:
        # birthplace: use the gold value directly
        return [row.gold]

todo['jgold'] = todo.apply(build_jgold, axis=1)
todo['jgold'] = todo.jgold.map(lambda g: list(dict.fromkeys(g)))

t = time.time()
with ThreadPoolExecutor(32) as ex:
    todo['verdict'] = list(ex.map(judge, zip(todo.relation, todo.jgold, todo.answer)))
print(len(todo), 'judged in', round(time.time() - t), 's', todo.verdict.value_counts().to_dict())
todo['judge_yes'] = todo.verdict.str.startswith('YES')

# BP_ACCEPT: hand-reviewed birthplace acceptances from the trace
BP_ACCEPT = {1232, 1293, 3714, 1291, 1452, 3230, 1444, 1642, 1628, 2851, 3228, 4453, 2848, 3003, 3233, 4464, 3197, 3469, 3458, 3892, 3895, 4253,
             5440, 5447, 2521, 2770, 5734, 5738, 1139, 1475, 1767, 2447, 4329, 5784, 2370, 2377, 581, 3724, 3733, 4694, 2214}

REF2 = re.compile(r"^\s*(none|n/a|not enough information|not available|information not available|unknown)\b", re.I)

bpy = todo[(todo.relation == 'BirthPlace') & todo.judge_yes]
bp_idx = set(bpy.index)
acc_keys = {(bpy.at[i, 'model'], bpy.at[i, 'qid']) for i in BP_ACCEPT if i in bpy.index}

T = todo.reset_index(drop=True)
T['final'] = T.judge_yes
bpmask = (T.relation == 'BirthPlace') & T.judge_yes
T.loc[bpmask, 'final'] = [(m, q) in acc_keys for m, q in zip(T[bpmask].model, T[bpmask].qid)]
T.loc[T.answer.str.match(REF2), 'final'] = False

J = T[['model', 'qid', 'person', 'relation', 'answer', 'jgold', 'verdict', 'judge_yes', 'final']].rename(
    columns={'jgold': 'wikidata_values', 'verdict': 'judge_verdict', 'final': 'accepted'})
J['hand_reviewed'] = (J.relation == 'BirthPlace') & J.judge_yes
J['wikidata_values'] = J.wikidata_values.map(lambda g: json.dumps(g, ensure_ascii=False))
J.to_csv('judge_verdicts.csv', index=False, encoding='utf-8-sig')
print(len(J), J.groupby(['relation', 'judge_yes', 'accepted']).size().to_dict())
