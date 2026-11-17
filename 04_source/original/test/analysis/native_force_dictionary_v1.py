"""Change force cadence only inside the named functionObject in a COPY."""
import re


def force_every_step(text):
    matches=list(re.finditer(r'(?m)^\s*obstacleForces\s*\{',text))
    if len(matches)!=1:raise ValueError('Exactly one obstacleForces block required')
    start=text.index('{',matches[0].start());depth=1;end=start+1
    while end<len(text) and depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    if depth:raise ValueError('Unclosed force dictionary')
    block=text[start:end]
    if len(re.findall(r'\bwriteControl\s+timeStep\s*;',block))!=1:raise ValueError('Expected timestep force control')
    block,n=re.subn(r'\bwriteInterval\s+10\s*;','writeInterval 1;',block)
    if n!=1:raise ValueError('Exactly one force cadence entry required')
    return text[:start]+block+text[end:]
