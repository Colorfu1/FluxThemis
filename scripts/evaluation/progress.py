"""Common stages and acknowledged episode progress for all benchmarks."""

import json
import re
import time

STAGES = ('读取配置', '仿真环境预检', '模型就绪检查', '仿真与评测', '结果核验')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Progress:

    def __init__(self):
        self.stage = 0
        self.started = time.monotonic()
        self.output = None

    def enter(self, stage):
        self.stage = stage
        self.started = time.monotonic()
        self.show('开始')

    def show(self, message, current=None, total=None, status='进行中'):
        elapsed = int(time.monotonic() - self.started)
        fraction = (f' {current}/{total} ({100 * current / total:.1f}%)'
                    if total else '')
        line = (
            f'[{self.stage}/{len(STAGES)} {STAGES[self.stage - 1]}][{status}]'
            f'{fraction} {message} | 本阶段 '
            f'{elapsed // 60:02d}:{elapsed % 60:02d}')
        print(line, flush=True)
        if self.output:
            with (self.output / 'progress.log').open('a') as stream:
                stream.write(line + '\n')
            write_json(
                self.output / 'progress.json',
                dict(
                    stage=self.stage,
                    title=STAGES[self.stage - 1],
                    status=status,
                    message=message,
                    current=current,
                    total=total,
                    stage_elapsed_s=elapsed))


class Events:

    def __init__(self):
        self.position = 0
        self.pending = ''
        self.sequence = 0
        self.session = None
        self.total = None
        self.episodes = {}
        self.active = {}
        self.episode_starts = {}
        self.restored = set()
        self.end = None
        self.run_dir = None

    def read(self, path):
        if not path.is_file():
            return
        with path.open() as stream:
            stream.seek(self.position)
            self.pending += stream.read()
            self.position = stream.tell()
        lines = self.pending.split('\n')
        self.pending = lines.pop()
        for line in lines:
            self.accept(json.loads(line))

    def accept(self, event):
        if event['sequence'] != self.sequence + 1:
            raise ValueError('本地事件序列不连续')
        session = event['run_session_id']
        if self.session is not None and self.session != session:
            raise ValueError('本地事件包含不同评测会话')
        self.sequence = event['sequence']
        self.session = event['run_session_id']
        self.run_dir = event['run_dir']
        kind, body = event['event_type'], event['payload']
        if kind == 'run_start':
            if self.total is not None:
                raise ValueError('重复的 run_start')
            self.total = body['total_episodes']
        elif kind == 'episode_start':
            self.active[body['episode_id']] = body['task_id']
            self.episode_starts[body['episode_id']] = body.get('started_at')
            if body.get('resumed_from'):
                self.restored.add(body['episode_id'])
        elif kind == 'episode_end':
            key = body['episode_id']
            if key in self.episodes:
                raise ValueError('重复的 episode_end')
            self.episodes[key] = body
            self.active.pop(key, None)
            if body['completed_episodes'] != len(self.episodes):
                raise ValueError('已完成计数与 episode 记录不一致')
        elif kind == 'run_end':
            self.end = body

    def result(self, exit_code):
        successes = sum(row['success'] for row in self.episodes.values())
        complete = bool(exit_code == 0 and self.end
                        and self.end['status'] == 'finished'
                        and self.total == len(self.episodes)
                        and self.end['completed_episodes'] == self.total
                        and self.end['total_episodes'] == self.total)
        if complete:
            overall = self.end['metrics']['overall']
            complete = (
                overall['episodes'] == self.total
                and overall['successes'] == successes)
        return dict(
            complete=complete,
            simulator_exit_code=exit_code,
            completed=len(self.episodes),
            total=self.total,
            successes=successes,
            remote_result_dir=self.run_dir,
            metrics=self.end.get('metrics') if self.end else None)


class SimulatorStartupLog:
    """Track current readiness and surface errors hidden in the raw log."""

    def __init__(self):
        self.ready = set()
        self.lines_seen = 0
        self.last_error = ''
        self.last_change = time.monotonic()

    def update(self, text):
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'):
            lines.pop()  # A writer may still be completing the final line.
        for line in lines[self.lines_seen:]:
            ready = re.search(r'simulator worker ready id=(\S+)', line)
            restart = re.search(
                r'restarting (simulator-worker-\S+) '
                r'after exit code (\S+)', line)
            if ready:
                self.ready.add(ready.group(1))
                self.last_change = time.monotonic()
            if restart:
                self.ready.discard(restart.group(1))
            if restart or re.search(
                    r'Traceback|Fatal Python error|(?:Error|Exception):',
                    line):
                self.last_error = line.strip()[-400:]
                print('[SIM_STARTUP_ERROR] ' + self.last_error, flush=True)
        self.lines_seen = len(lines)
        idle = int(time.monotonic() - self.last_change)
        message = f'；就绪数 {idle}s 未增加' if idle >= 60 else ''
        if self.last_error:
            message += '；最近异常：' + self.last_error
        return message
