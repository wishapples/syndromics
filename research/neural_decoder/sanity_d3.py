import chip, numpy as np, pymatching
real,_=chip.real_chip_circuit(3,3,0.004); nom=chip.nominal_circuit(3,3,0.004)
def lut_ler(circ, ntr=20_000_000, nte=500_000):
    w=(1<<np.arange(24)).astype(np.int64)
    cnt=np.zeros((1<<24,2),np.int32)
    for s in range(10):
        X,y=chip.sample(circ,ntr//10,seed=100+s); k=X.astype(np.int64)@w
        np.add.at(cnt,(k,y),1)
    Xt,yt=chip.sample(circ,nte,seed=2); kt=Xt.astype(np.int64)@w
    pr=(cnt[kt,1]>cnt[kt,0]).astype(np.uint8)
    m=pymatching.Matching.from_detector_error_model(nom.detector_error_model(decompose_errors=True))
    return (pr!=yt).mean(), (m.decode_batch(Xt)[:,0]!=yt).mean()
print("REAL chip   : LUT(near-optimal) %.4f  MWPM-uniform %.4f"%lut_ler(real))
print("NOMINAL chip: LUT(near-optimal) %.4f  MWPM-uniform %.4f"%lut_ler(nom))
