"""In-memory Q4 observation environment. Generated truths never enter the client."""
import copy, hashlib, math

class ObservationClient:
    __slots__=('act','position','channel','virtual','rows')
    def __init__(self,action):
        self.act=action;self.position=(0.,0.);self.channel=1;self.virtual=0.;self.rows=[]

class World:
    def __init__(self,case):
        self.case=copy.deepcopy(case);self.sources={s['channel']:s for s in self.case['sources']};self.cleared=set()
        self.client=ObservationClient(self._act);self.entered=False;self.finished=False
    def _error(self,position,channel):
        mode=self.case['error_mode']
        if mode=='plus':return 1.
        if mode=='minus':return -1.
        if mode=='zero':return 0.
        if mode=='sine':return math.sin(position[0]*.011+position[1]*.017+channel*.731)
        key=f"{self.case['seed']}/{position[0]:.8f}/{position[1]:.8f}/{channel}".encode()
        return 2*int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/(2**64-1)-1
    def _act(self,path,position=None,channel=None):
        c=self.client;before=c.virtual;movement=0.;switch=0;operation=0.
        if path=='/enter':
            if self.entered:raise RuntimeError('Enter twice')
            self.entered=True;reply={'status':'success'}
        elif path=='/exit':
            self.finished=True;reply={'status':'success','exit_reason':'user_exit'}
        else:
            if not self.entered or self.finished:raise RuntimeError('No active offline session')
            if path not in ('/measure','/clear'):raise ValueError(path)
            point=tuple(map(float,position));channel=int(channel)
            if len(point)!=2 or not all(math.isfinite(x) and abs(x)<=2e6 for x in point) or not 1<=channel<=20:raise ValueError('Invalid action')
            movement=math.dist(c.position,point)/5;c.position=point;s=self.sources.get(channel)
            present=s is not None and channel not in self.cleared
            distance=math.dist(point,s['position']) if present else math.inf
            if path=='/measure':
                switch=int(channel!=c.channel);c.channel=channel;operation=5.
                visible=False
                if present and distance<=s['radius']+1e-10:
                    if s['type']=='omni':visible=True
                    else:
                        ang=math.radians(s['direction_deg']);dx=point[0]-s['position'][0];dy=point[1]-s['position'][1]
                        visible=math.cos(ang)*dx+math.sin(ang)*dy>=-1e-10
                if not visible:reply={'measure_result':'no_signal'}
                elif distance<=5+1e-10:reply={'measure_result':'near'}
                else:
                    angle=math.degrees(math.atan2(s['position'][1]-point[1],s['position'][0]-point[0]))
                    reply={'measure_result':'direction','svd_deg':round((angle+self._error(point,channel))%360,2)%360}
            else:
                success=present and distance<=20+1e-10
                operation=5. if success else 3.
                if success:self.cleared.add(channel)
                reply={'clear_result':'success' if success else 'no_target_in_range'}
            c.virtual+=movement+switch+operation
        c.rows.append({'sequence':len(c.rows)+1,'path':path,'position':list(position) if position is not None else None,
            'channel':channel,'response':reply.copy(),'movement_s':movement,'switch_s':switch,'operation_s':operation,
            'virtual_before_s':before,'virtual_after_s':c.virtual})
        return reply
