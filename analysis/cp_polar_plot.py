#!/usr/bin/env python3
# Cp-on-normals normalized plot from a Tecplot file
# Usage: python make_cp_normals.py /path/to/soln.dat [scale_fraction=0.07]
import sys, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TecplotZone:
    def __init__(self, title, attrs):
        self.title = title
        self.attrs = attrs
        self.nodes = None
        self.elem = None

def parse_variables(header_text):
    m = re.search(r'VARIABLES\s*=\s*(.+?)(?:\nZONE|\nTITLE|$)', header_text, flags=re.S|re.I)
    varblob = m.group(1)
    vars_quoted = re.findall(r'"([^"]+)"', varblob)
    return [v.strip() for v in vars_quoted] if vars_quoted else [t for t in re.split(r'[,\s]+', varblob.strip()) if t]

def tokenize_numbers(text, count=None):
    num_pat = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?'
    if count is None:
        nums = [float(x.replace('D','E').replace('d','e')) for x in re.findall(num_pat, text)]
        return np.array(nums, dtype=float), ''
    it = re.finditer(num_pat, text)
    vals = []
    endpos = 0
    for k, m in enumerate(it):
        vals.append(float(m.group(0).replace('D','E').replace('d','e')))
        endpos = m.end()
        if k+1 == count:
            break
    return np.array(vals, dtype=float), text[endpos:]

def parse_zone_attrs(line):
    title = None
    attrs = {}
    for key, valq, valu in re.findall(r'(\w+)\s*=\s*(?:"([^"]+)"|([^\s,]+))', line):
        val = valq if valq not in (None, '') else valu
        if key.upper() in ('T','TITLE'): title = (val or '').strip()
        else: attrs[key.upper()] = (val or '').strip()
    if title is None:
        m = re.search(r'ZONE\s+([^,]+)', line, flags=re.I)
        if m: title = m.group(1).strip()
    return title or '', attrs

def expected_nodes_per_element(zonetype):
    return {'FELINESEG':2,'FETRIANGLE':3,'FEQUADRILATERAL':4,'FEPOLYGON':-1,'FETETRAHEDRON':4,'FEBRICK':8}.get((zonetype or '').upper(),-1)

def parse_tecplot(filepath: Path):
    text = filepath.read_text(errors='ignore')
    zones_splits = re.split(r'(?i)(?=^\s*ZONE\b)', text, flags=re.M)
    header = zones_splits[0]
    variables = parse_variables(header)
    zones = []
    remaining_text = text[len(header):]
    zone_iter = list(re.finditer(r'(?im)^\s*ZONE\b.*$', remaining_text))
    end_positions = [m.end() for m in zone_iter] + [len(remaining_text)]
    for idx, m in enumerate(zone_iter):
        zone_line = m.group(0)
        z_title, attrs = parse_zone_attrs(zone_line)
        z = TecplotZone(z_title, attrs)
        start = end_positions[idx]
        stop = end_positions[idx+1] if idx+1 < len(end_positions) else len(remaining_text)
        zone_body = remaining_text[start:stop]
        N = int(attrs.get('NODES') or attrs.get('N') or 0)
        E = int(attrs.get('ELEMENTS') or attrs.get('E') or 0)
        zonetype = attrs.get('ZONETYPE','').upper()
        datapack = attrs.get('DATAPACKING','POINT').upper()
        nnpe = expected_nodes_per_element(zonetype)
        if N > 0:
            if datapack == 'POINT':
                vals, rem = tokenize_numbers(zone_body, N*len(variables)); z.nodes = vals.reshape(N, len(variables)); zone_body = rem
            else:
                data = np.empty((len(variables), N), dtype=float); rem = zone_body
                for vi in range(len(variables)):
                    vals, rem = tokenize_numbers(rem, N); data[vi,:] = vals
                z.nodes = data.T; zone_body = rem
        if E > 0 and nnpe != -1:
            conn, rem2 = tokenize_numbers(zone_body, E*nnpe); z.elem = conn.astype(int).reshape(E, nnpe) - 1
        elif E > 0 and nnpe == -1:
            try: conn, rem2 = tokenize_numbers(zone_body, E*2); z.elem = conn.astype(int).reshape(E, 2) - 1
            except Exception: z.elem=None
        zones.append(z)
    return variables, zones

def find_var_index(varnames, candidates):
    lower = [v.lower() for v in varnames]
    for cand in candidates:
        for i, v in enumerate(lower):
            if v == cand.lower() or cand.lower() in v: return i
    return None

def order_points_from_lineseg(npts, elements):
    adj = {i: [] for i in range(npts)}
    for a,b in elements: a=int(a); b=int(b); adj[a].append(b); adj[b].append(a)
    endpoints=[i for i,n in adj.items() if len(n)==1]; start=endpoints[0] if endpoints else 0
    order=[start]; vis={start}; cur=start
    while True:
        nxt=[n for n in adj[cur] if n not in vis]
        if not nxt: break
        cur=nxt[0]; order.append(cur); vis.add(cur)
    if len(order)<npts: order.extend([i for i in range(npts) if i not in vis])
    return order

def boundary_loop_order(npts, elements):
    edge_count={}
    for elem in elements:
        for a,b in zip(elem, np.roll(elem,-1)):
            a=int(a); b=int(b); e=(min(a,b),max(a,b)); edge_count[e]=edge_count.get(e,0)+1
    boundary=[e for e,c in edge_count.items() if c==1]
    if not boundary: return []
    adj={i:[] for i in range(npts)}; bnodes=set()
    for a,b in boundary: adj[a].append(b); adj[b].append(a); bnodes.add(a); bnodes.add(b)
    endpoints=[i for i in bnodes if len(adj[i])==1]; start=endpoints[0] if endpoints else min(bnodes)
    order=[start]; vis={start}; cur=start; prev=None
    while True:
        nbrs=adj[cur]; nxts=[n for n in nbrs if n!=prev]
        if not nxts: break
        nxt=nxts[0]
        if nxt in vis:
            if nxt==order[0] and len(vis)==len(bnodes): break
            alt=[n for n in nbrs if n not in (prev,nxt)]
            if alt: nxt=alt[0]
            else: break
        order.append(nxt); vis.add(nxt); prev, cur = cur, nxt
        if len(vis)>=len(bnodes) and order[0] in adj[cur]: break
    return [i for i in order if i in bnodes]

def nearest_neighbor_order(xy):
    n=xy.shape[0]
    if n==0: return []
    unused=set(range(n)); start=int(np.argmin(xy[:,0])); order=[start]; unused.remove(start); last=start
    while unused:
        idxs=np.array(sorted(list(unused))); d=np.linalg.norm(xy[idxs]-xy[last],axis=1)
        j=idxs[int(np.argmin(d))]; order.append(int(j)); unused.remove(int(j)); last=int(j)
    return order

def order_zone(z, x_idx, y_idx):
    X = z.nodes[:, x_idx]; Y = z.nodes[:, y_idx]
    if z.elem is not None:
        if z.elem.shape[1]==2: ord_idx = order_points_from_lineseg(len(X), z.elem)
        else:
            ord_idx = boundary_loop_order(len(X), z.elem)
            if not ord_idx: ord_idx = nearest_neighbor_order(np.column_stack([X,Y]))
    else:
        ord_idx = nearest_neighbor_order(np.column_stack([X,Y]))
    return np.array(ord_idx, dtype=int)

def compute_freestream(varnames, zones, wall_zone_idxs, x_idx, y_idx, p_idx):
    rho_idx = find_var_index(varnames, ['Density','rho'])
    u_idx = find_var_index(varnames, ['U','u-velocity','u_velocity'])
    v_idx = find_var_index(varnames, ['V','v-velocity','v_velocity'])
    w_idx = find_var_index(varnames, ['W','w-velocity','w_velocity'])
    Xs=[]; Ys=[]; Ps=[]; Rhos=[]; Speeds=[]
    for zi,z in enumerate(zones):
        if zi in wall_zone_idxs or z.nodes is None: continue
        Xs.append(z.nodes[:, x_idx]); Ys.append(z.nodes[:, y_idx])
        U=z.nodes[:,u_idx]; V=z.nodes[:,v_idx]; W=z.nodes[:, w_idx] if w_idx is not None and w_idx<z.nodes.shape[1] else 0.0
        Vmag=np.sqrt(U*U+V*V+(W if np.isscalar(W) else W)**2)
        Ps.append(z.nodes[:, p_idx]); Rhos.append(z.nodes[:, rho_idx]); Speeds.append(Vmag)
    X=np.concatenate(Xs); Y=np.concatenate(Ys); p=np.concatenate(Ps); rho=np.concatenate(Rhos); Vmag=np.concatenate(Speeds)
    bodyX=[]; bodyY=[]
    for zi in wall_zone_idxs:
        bodyX.append(zones[zi].nodes[:, x_idx]); bodyY.append(zones[zi].nodes[:, y_idx])
    cx, cy = float(np.mean(np.concatenate(bodyX))), float(np.mean(np.concatenate(bodyY)))
    r2=(X-cx)**2+(Y-cy)**2; mask=r2>=np.quantile(r2,0.90)
    p_inf=float(np.median(p[mask])); rho_inf=float(np.median(rho[mask])); U_inf=float(np.median(Vmag[mask]))
    q_inf=0.5*rho_inf*U_inf**2
    return p_inf, q_inf

def ensure_increasing(seq, X):
    seq=list(map(int,seq))
    if X[seq[0]] > X[seq[-1]]: seq = list(reversed(seq))
    return seq

def build_segments(z0, z1, x_idx, y_idx):
    X0=z0.nodes[:, x_idx]; X1=z1.nodes[:, x_idx]
    ord0=order_zone(z0, x_idx, y_idx); ord1=order_zone(z1, x_idx, y_idx)
    seam_x=0.5*(float(np.max(X1))+float(np.min(X0)))
    # open split (z0)
    X=z0.nodes[:, x_idx]; Y=z0.nodes[:, y_idx]
    d=np.abs(X-seam_x); cand=np.argsort(d)[:12]
    i_top=int(cand[np.argmax(Y[cand])]); i_bot=int(cand[np.argmin(Y[cand])])
    pos={int(n):k for k,n in enumerate(ord0)}; tpos,bpos=pos[i_top],pos[i_bot]
    arc_t=ord0[tpos:]; arc_b=ord0[bpos:]
    if np.mean(Y[arc_t[:min(10,len(arc_t))]]) >= np.mean(Y[arc_b[:min(10,len(arc_b))]]): upper0, lower0 = arc_t, arc_b
    else: upper0, lower0 = arc_b, arc_t
    upper0=ensure_increasing(upper0, X0); lower0=ensure_increasing(lower0, X0)
    # closed split (z1)
    X=z1.nodes[:, x_idx]; Y=z1.nodes[:, y_idx]
    d=np.abs(X-seam_x); cand=np.argsort(d)[:12]
    i_top=int(cand[np.argmax(Y[cand])]); i_bot=int(cand[np.argmin(Y[cand])])
    pos={int(n):k for k,n in enumerate(ord1)}; tpos,bpos=pos[i_top],pos[i_bot]
    if tpos<=bpos:
        arc_f=ord1[tpos:bpos+1]; arc_w=np.concatenate([ord1[bpos:],ord1[:tpos+1]])
    else:
        arc_f=ord1[bpos:tpos+1]; arc_w=np.concatenate([ord1[tpos:],ord1[:bpos+1]])
    if np.mean(Y[arc_f])>=np.mean(Y[arc_w]): upper1, lower1 = arc_f, arc_w
    else: upper1, lower1 = arc_w, arc_f
    upper1=ensure_increasing(upper1, X1); lower1=ensure_increasing(lower1, X1)
    return [(0, list(reversed(upper0))), (1, list(reversed(upper1))), (1, list(lower1)), (0, list(lower0))]

def merge_segments_to_arrays(segments, z0, z1, x_idx, y_idx, p_idx, tol_geom=1e-6):
    X_all=[]; Y_all=[]; P_all=[]
    for zid, idxs in segments:
        Xz = z0.nodes[:, x_idx] if zid==0 else z1.nodes[:, x_idx]
        Yz = z0.nodes[:, y_idx] if zid==0 else z1.nodes[:, y_idx]
        Pz = z0.nodes[:, p_idx] if zid==0 else z1.nodes[:, p_idx]
        for i in idxs:
            x, y, p = float(Xz[i]), float(Yz[i]), float(Pz[i])
            if X_all and ((X_all[-1]-x)**2 + (Y_all[-1]-y)**2)**0.5 < tol_geom:
                continue
            X_all.append(x); Y_all.append(y); P_all.append(p)
    return np.array(X_all), np.array(Y_all), np.array(P_all)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python make_cp_normals.py /path/to/soln.dat [scale_fraction=0.07]')
        sys.exit(1)
    tecfile = Path(sys.argv[1])
    scale_fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.07
    varnames, zones = parse_tecplot(tecfile)
    x_idx = find_var_index(varnames, ['X']); y_idx = find_var_index(varnames, ['Y']); p_idx = find_var_index(varnames, ['Pressure','StaticPressure','p'])
    wall_zone_idxs = [i for i,z in enumerate(zones) if re.search(r'wall', z.title, flags=re.I)]
    if not wall_zone_idxs: wall_zone_idxs=[i for i,z in enumerate(zones) if z.elem is not None]
    wall_zone_idxs = wall_zone_idxs[:2]; z0, z1 = zones[wall_zone_idxs[0]], zones[wall_zone_idxs[1]]
    p_inf, q_inf = compute_freestream(varnames, zones, wall_zone_idxs, x_idx, y_idx, p_idx)
    segments = build_segments(z0, z1, x_idx, y_idx)
    Xw, Yw, Pw = merge_segments_to_arrays(segments, z0, z1, x_idx, y_idx, p_idx, tol_geom=1e-6)
    Cp = (Pw - p_inf)/(q_inf if q_inf!=0 else 1.0)
    cx, cy = np.mean(Xw), np.mean(Yw)
    pts = np.column_stack([Xw, Yw])
    t = np.zeros_like(pts); t[1:-1] = pts[2:] - pts[:-2]; t[0] = pts[1]-pts[0]; t[-1] = pts[-1]-pts[-2]
    tn=np.linalg.norm(t,axis=1,keepdims=True); tn[tn==0]=1.0; t=t/tn
    n = np.column_stack([-t[:,1], t[:,0]]); vec=pts-np.array([[cx,cy]]); sgn=np.sign(np.sum(vec*n,axis=1,keepdims=True)); sgn[sgn==0]=1.0; n=n*sgn
    c = float(np.max(Xw)-np.min(Xw))
    Cp_norm = Cp / (np.max(np.abs(Cp)) if np.max(np.abs(Cp))>0 else 1.0)
    scale = scale_fraction * c
    dX = n[:,0]*(Cp_norm*scale); dY = n[:,1]*(Cp_norm*scale)
    plt.figure()
    plt.fill_between(Xw, Yw, Yw, alpha=0.2)
    for (x,y,dx,dy) in zip(Xw, Yw, dX, dY):
        plt.plot([x, x+dx], [y, y+dy])
    plt.axis('equal'); plt.title('Cp entlang Normale (normalisiert)'); plt.xlabel('x'); plt.ylabel('y')
    plt.savefig('cp_normals.png', dpi=200, bbox_inches='tight')
    print('Saved cp_normals.png')
