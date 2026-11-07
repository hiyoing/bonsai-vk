import numpy as np, wgpu, importlib
p = importlib.import_module("probe_gemv15")
M, K = 8, 128; nb = K // 128; rng = np.random.default_rng(7)
words, tr, d = p.make_blocks(M, K, nb, rng)
a = rng.integers(0, 256, size=K, dtype=np.uint8)
x = (a.astype(np.float32) / 255.0).astype(np.float32)
def elem_of(k,t,b): return (4*k+16*t if k<4 else 80+4*(k-4)+8*t)+b
xs = x.reshape(nb,128)
bs = np.zeros((nb,4), np.float32)
for b in range(nb):
    for h in range(2):
        idx=[elem_of(3*h+kk,t,bb) for kk in range(3) for t in range(5) for bb in range(4)]
        if h==0: idx += list(range(120,128))
        bs[b,h]=xs[b][idx].sum()
    for sub in range(2):
        idx=[elem_of(3*sub+kk,t,bb) for kk in range(3) for t in range(5) for bb in range(4)]
        if sub==1: idx += list(range(120,128))
        bs[b,2+sub]=xs[b][idx].sum()
w1=(tr.astype(np.float32)-1.0)
ref=np.zeros(M,np.float64)
for b in range(nb): ref += d[b].astype(np.float64)*(w1[:,b,:]@xs[b].astype(np.float64))
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
ents=[]
for i in range(9):
    t=(wgpu.BufferBindingType.storage if i==4 else wgpu.BufferBindingType.uniform if i==5 else wgpu.BufferBindingType.read_only_storage)
    ents.append({"binding":i,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":t}})
bgl=dev.create_bind_group_layout(entries=ents); pl=dev.create_pipeline_layout(bind_group_layouts=[bgl])
def buf(arr,usage=wgpu.BufferUsage.STORAGE): return dev.create_buffer_with_data(data=arr.tobytes(),usage=usage)
pad=np.zeros((K//4+3)//4*4,np.uint32); pad[:K//4]=a.view(np.uint32)
src=p.vec_src(1)
pipe=dev.create_compute_pipeline(layout=pl,compute={"module":dev.create_shader_module(code=src),"entry_point":"main"})
bp=buf(np.array([M,K,nb,1],np.uint32),wgpu.BufferUsage.UNIFORM)
bo=dev.create_buffer(size=M*4,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
sz=lambda b: b.nbytes
bg=dev.create_bind_group(layout=bgl,entries=[
 {"binding":0,"resource":{"buffer":buf(words),"offset":0,"size":words.nbytes}},
 {"binding":1,"resource":{"buffer":buf(x),"offset":0,"size":x.nbytes}},
 {"binding":2,"resource":{"buffer":buf(d.reshape(-1)),"offset":0,"size":d.nbytes}},
 {"binding":3,"resource":{"buffer":buf(bs.reshape(-1)),"offset":0,"size":bs.nbytes}},
 {"binding":4,"resource":{"buffer":bo,"offset":0,"size":M*4}},
 {"binding":5,"resource":{"buffer":bp,"offset":0,"size":16}},
 {"binding":6,"resource":{"buffer":buf(pad),"offset":0,"size":pad.nbytes}},
 {"binding":7,"resource":{"buffer":buf(pad),"offset":0,"size":pad.nbytes}},
 {"binding":8,"resource":{"buffer":buf((np.ones(nb,np.float32)/255)),"offset":0,"size":nb*4}}])
e=dev.create_command_encoder(); c=e.begin_compute_pass(); c.set_pipeline(pipe); c.set_bind_group(0,bg); c.dispatch_workgroups(M,1,1); c.end(); dev.queue.submit([e.finish()])
got=np.frombuffer(dev.queue.read_buffer(bo,0,M*4),np.float32)
print("ref ", np.round(ref,4)); print("got ", got); print("relerr", abs(got-ref).max()/abs(ref).max())
