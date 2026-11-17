"""Sequential configurators and auditable time/attempt accounting, no dataset I/O."""
import math
import os
import random
import signal
import subprocess
import time


FLOATS={'learning_rate':(1e-4,3e-3),'weight_decay':(1e-6,1e-2)}
CATEGORIES={'pressure_weight':[.25,.5,1,2,4],'divergence_weight':[0,.001,.003,.01,.03]}
KEYS=tuple(FLOATS)+tuple(CATEGORIES)


def validate_parameters(params):
    if set(params)!=set(KEYS):raise ValueError('Exactly four hyperparameters required')
    for key,(low,high) in FLOATS.items():
        value=params[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid numeric parameter: '+key)
    for key,choices in CATEGORIES.items():
        if isinstance(params[key],bool) or params[key] not in choices:raise ValueError('Invalid category: '+key)


class RandomConfigurator:
    def __init__(self,seed):self.rng=random.Random(seed)
    def ask(self):
        values={k:math.exp(self.rng.uniform(math.log(lo),math.log(hi))) for k,(lo,hi) in FLOATS.items()}
        values.update({k:self.rng.choice(v) for k,v in CATEGORIES.items()})
        validate_parameters(values);return values


class TPEConfigurator:
    def __init__(self,seed,history):
        import optuna
        self.optuna=optuna
        self.distributions={k:optuna.distributions.FloatDistribution(lo,hi,log=True) for k,(lo,hi) in FLOATS.items()}
        self.distributions.update({k:optuna.distributions.CategoricalDistribution(v) for k,v in CATEGORIES.items()})
        self.sampler_settings={'seed':seed,'n_startup_trials':3,'n_ei_candidates':24,'multivariate':False,'constant_liar':False}
        self.study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(**self.sampler_settings),
                                      pruner=optuna.pruners.NopPruner())
        self.pending=None
        for row in history:
            validate_parameters(row['parameters'])
            score=row['S']
            if not math.isfinite(score):raise ValueError('Finite initial history required')
            self.study.add_trial(optuna.trial.create_trial(params=row['parameters'],distributions=self.distributions,value=score))
    def ask(self):
        if self.pending is not None:raise RuntimeError('Previous trial must be told before next ask')
        trial=self.study.ask(fixed_distributions=self.distributions)
        self.pending=trial
        validate_parameters(trial.params);return dict(trial.params)
    def tell(self,score=None,failed=False):
        if self.pending is None:raise RuntimeError('No pending trial')
        if failed:value=float('inf')
        else:
            if score is None or not math.isfinite(score):raise ValueError('Finite success score required')
            value=score
        self.study.tell(self.pending,value);self.pending=None
    def export(self):
        return {'optuna':self.optuna.__version__,'sampler_settings':self.sampler_settings,
                'trials':[{'number':t.number,'state':t.state.name,'parameters':t.params,
                           'S':t.value if t.value is not None and math.isfinite(t.value) else None,
                           'positive_infinity_penalty':t.value==float('inf')} for t in self.study.trials]}


class TrainingBudget:
    """Training-process wall seconds on one GPU; model decision overhead separate."""
    def __init__(self,total_seconds,max_attempts,shared_seconds=0,shared_attempts=0):
        if not all(math.isfinite(v) and v>=0 for v in (total_seconds,shared_seconds)) or total_seconds<=0:
            raise ValueError('Invalid time budget')
        if type(max_attempts) is not int or max_attempts<1 or type(shared_attempts) is not int or not 0<=shared_attempts<=max_attempts:
            raise ValueError('Invalid attempt budget')
        self.total_seconds=total_seconds;self.max_attempts=max_attempts
        self.spent_seconds=shared_seconds;self.attempts=shared_attempts;self.pending=None
        self.events=[{'type':'shared_history','seconds':shared_seconds,'attempts':shared_attempts}]
    @property
    def remaining_seconds(self):return max(0,self.total_seconds-self.spent_seconds)
    def can_start(self,minimum_seconds=0):
        return self.pending is None and self.attempts<self.max_attempts and self.remaining_seconds>minimum_seconds
    def begin(self,label,minimum_seconds=0):
        if not self.can_start(minimum_seconds):raise RuntimeError('Attempt or time budget exhausted')
        self.attempts+=1;self.pending=label;return self.remaining_seconds
    def finish(self,label,seconds,score=None,status='failed'):
        if label!=self.pending:raise RuntimeError('No matching pending attempt')
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid measured duration')
        if status=='complete' and (score is None or not math.isfinite(score)):raise ValueError('Finite completed score required')
        self.spent_seconds+=seconds
        self.events.append({'type':'attempt','label':label,'status':status,'seconds':seconds,
                            'S':score if status=='complete' else None,'positive_infinity_penalty':status!='complete'})
        self.pending=None
    def export(self):
        return {'training_seconds_limit':self.total_seconds,'attempt_limit':self.max_attempts,
                'training_seconds_spent':self.spent_seconds,'attempts_spent':self.attempts,
                'remaining_seconds':self.remaining_seconds,'overrun_seconds':max(0,self.spent_seconds-self.total_seconds),
                'pending':self.pending,'events':self.events}


def run_bounded_process(command,seconds,log_path,cleanup_seconds=2):
    """Kill only this command's process group. Keep measured cleanup cost visible."""
    if not math.isfinite(seconds) or seconds<=cleanup_seconds or cleanup_seconds<=0:
        raise ValueError('Insufficient room for bounded cleanup')
    started=time.perf_counter();timed_out=False
    with open(log_path,'w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=max(.001,seconds-cleanup_seconds-(time.perf_counter()-started)))
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=cleanup_seconds/2)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:process.wait(timeout=cleanup_seconds/2)
                except subprocess.TimeoutExpired:raise RuntimeError('Own process group did not finish cleanup; stop controller')
    return {'exit_code':process.returncode,'timed_out':timed_out,'wall_seconds':time.perf_counter()-started}
