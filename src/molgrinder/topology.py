import math

import numpy as np
from rdkit import Chem
from scipy.spatial import ConvexHull, QhullError

'''
This code is based on 10.1021/acs.jctc.1c00322
with slight modifications for Martini 3 compatibility
'''

MARTINI_DIH_K = 500.0
MARTINI_ANGLE_K = 25.0
MARTINI_BOND_K = 1250.0
MARTINI_VS_MASS = 0
MARTINI_MASS = 72.0
MARTINI_S_MASS = 54.0
MARTINI_T_MASS = 36.0
MARTINI_RING_MIN_K = 5000000

def bead_coords(bead, conf, mol):
    """Center of geometry of the heavy atoms in one bead, in nm.

    Hydrogens are left out. Martini maps the heavy-atom group, and an
    explicit hydrogen pulls a methyl or hydroxyl bead off that center.
    """
    coords = np.zeros(3, dtype=float)
    num_atoms = 0
    for atom in bead:
        atom = int(atom)
        if mol.GetAtomWithIdx(atom).GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom)
        coords += (pos.x, pos.y, pos.z)
        num_atoms += 1
    if num_atoms == 0:
        raise RuntimeError("bead has no heavy atoms to place")
    return coords / (num_atoms * 10.0)

def _centered_box(coords0):
    """Put the bead center of geometry in the middle of a box that fits it.

    One nanometre of padding on each side keeps beads off the periodic boundary.
    Coordinates are returned in nm; the box vector is in nm.
    """
    coords = np.asarray(coords0, dtype=float)
    span = coords.max(axis=0) - coords.min(axis=0)
    box = span + 2.0
    center = coords.mean(axis=0)
    shifted = coords - center + box / 2.0
    return shifted, box


def write_gro(mol, mol_name, bead_names, coords0, A_cg, name_CG, name_AA = None):
    #write gro file
    placed, box = _centered_box(coords0)
    with open(name_CG + '_CG.gro', 'w') as gro:
        gro.write('single molecule of {}\n'.format(mol_name))
        gro.write('{}\n'.format(len(bead_names)))
        i = 1
        for bead, xyz in zip(bead_names, placed):
            gro.write('{:5d}{:5}{:>5}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(1, mol_name, bead, i, xyz[0], xyz[1], xyz[2]))
            i += 1
        gro.write('{0:10.5f}{1:10.5f}{2:10.5f}\n'.format(box[0], box[1], box[2]))

    with open(name_CG+'_CG.pdb', 'w') as pdb:
        i = 1
        for bead, xyz in zip(bead_names, placed):
            pdb.write(
            "HETATM{serial:5d} {name:<4s} {resname:>3s} {chain:1s}{resid:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}\n"
                .format(
                    serial=i,
                    name=str(bead)[:4],
                    resname=str(mol_name)[:3],
                    chain='A',
                    resid=1,
                    x=xyz[0]*10.0, y=xyz[1]*10.0, z=xyz[2]*10.0,  # drop *10.0 if xyz already in Å
                    occ=1.00,
                    bfac=0.00,
                    elem='B'
                )
            )
            i += 1
            
        # add bonds to pdb
        n = A_cg.shape[0]
        bonds = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if A_cg[i, j] != 0:
                    bonds[i].append(j)
                    bonds[j].append(i)

        atom_serial_start=1
        for i in range(n):
            si = atom_serial_start + i
            nbrs = [atom_serial_start + j for j in bonds[i]]
            for k in range(0, len(nbrs), 4):
                chunk = nbrs[k:k + 4]
                pdb.write("CONECT{:5d}".format(si) + "".join("{:5d}".format(x) for x in chunk) + "\n")
                

    if name_AA:
        Chem.MolToPDBFile(mol, name_AA + '_AA.pdb')
    


# A barycentric weight below this is numerical noise, not a bead outside the triangle.
_WEIGHT_TOL = 1e-8


def _project_ring(ring, coords):
    """Project ring-bead coordinates onto the plane of the two smallest inertia axes."""
    coords_r = np.array([coords[a] for a in ring], dtype=float)
    coords_c = coords_r - coords_r.mean(axis=0)
    x, y, z = coords_c.T
    inertia = np.array(
        [
            [np.sum(y * y + z * z), -np.sum(x * y), -np.sum(x * z)],
            [-np.sum(x * y), np.sum(x * x + z * z), -np.sum(y * z)],
            [-np.sum(x * z), -np.sum(y * z), np.sum(x * x + y * y)],
        ],
        dtype=float,
    )
    _vals, vecs = np.linalg.eigh(inertia)
    return np.column_stack([coords_c @ vecs[:, 0], coords_c @ vecs[:, 1]])


def _insert_on_closest_edge(real_sites, vs, coords):
    """Place vs into the real-site cycle, between the edge it lies closest to."""
    point = np.asarray(coords[vs], dtype=float)
    closest = 0
    best = float("inf")
    for edge in range(len(real_sites)):
        start = np.asarray(coords[real_sites[edge]], dtype=float)
        end = np.asarray(coords[real_sites[(edge + 1) % len(real_sites)]], dtype=float)
        span = end - start
        length2 = float(np.dot(span, span))
        if length2 < 1e-16:
            dist = float(np.linalg.norm(point - start))
        else:
            projected = start + (np.dot(span, point - start) / length2) * span
            dist = float(np.linalg.norm(point - projected))
        if dist < best:
            closest = edge
            best = dist
    real_sites.insert((closest + 1) % len(real_sites), vs)


def _barycentric(point, a, b, c):
    """Weights (wa, wb, wc) such that point = wa*a + wb*b + wc*c and the weights sum to 1.

    Returns None when the three points do not form a triangle.
    """
    matrix = np.array(
        [[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]],
        dtype=float,
    )
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    try:
        solved = np.linalg.solve(matrix, np.asarray(point, dtype=float) - a)
    except np.linalg.LinAlgError:
        return None
    weight_b = float(solved[0])
    weight_c = float(solved[1])
    return 1.0 - weight_b - weight_c, weight_b, weight_c


def _containing_triangle(vs_local, real_sites, coords_p, ring):
    """Barycentric weights of a real-site triangle that contains the projected bead.

    Triangles are tried nearest-first. A weight below -_WEIGHT_TOL means the
    bead is outside that triangle, and the weight is not clamped. None means
    no triangle contains the bead.
    """
    from itertools import combinations

    point = coords_p[vs_local]
    local = [ring.index(site) for site in real_sites]
    if len(local) < 3:
        return None
    distances = [float(np.linalg.norm(coords_p[idx] - point)) for idx in local]
    nearest_first = sorted(range(len(local)), key=lambda idx: distances[idx])
    rank = {idx: place for place, idx in enumerate(nearest_first)}
    combos = list(combinations(range(len(local)), 3))
    combos.sort(key=lambda comb: sum(rank[idx] for idx in comb))
    for i, j, k in combos:
        weights = _barycentric(
            point, coords_p[local[i]], coords_p[local[j]], coords_p[local[k]]
        )
        if weights is None or min(weights) < -_WEIGHT_TOL:
            continue
        cleaned = [0.0 if weight < 0.0 else weight for weight in weights]
        total = sum(cleaned)
        if total <= 0.0:
            continue
        cleaned = [weight / total for weight in cleaned]
        return {
            real_sites[i]: cleaned[0],
            real_sites[j]: cleaned[1],
            real_sites[k]: cleaned[2],
        }
    return None


def get_virtual_sites(ring, coords, A_cg):
    """Split a ring into real beads and virtual sites.

    Real beads are the convex hull, plus any inner bead that bonds outside the
    ring or whose projection is not inside a triangle of those real beads.
    Virtual sites are the remaining inner beads, each written as a barycentric
    combination of one containing triangle.
    """
    ring = [int(site) for site in ring]
    if len(ring) <= 3:
        return list(ring), {}

    try:
        coords_p = _project_ring(ring, coords)
        hull = ConvexHull(coords_p)
    except QhullError as exc:
        print(f"ring projection is degenerate ({exc}); keeping {len(ring)} beads real")
        return list(ring), {}

    real_sites = [ring[int(vertex)] for vertex in hull.vertices]
    virtual_sites = [site for site in ring if site not in real_sites]

    def promote(vs):
        virtual_sites.remove(vs)
        _insert_on_closest_edge(real_sites, vs, coords)

    for vs in list(virtual_sites):
        bonded_outside = any(int(other) not in ring for other in np.nonzero(A_cg[vs])[0])
        if bonded_outside:
            promote(vs)

    vs_weights = {}
    for vs in list(virtual_sites):
        weights = _containing_triangle(ring.index(vs), real_sites, coords_p, ring)
        if weights is None:
            print(f"bead {vs} is outside the ring triangles; keeping it as a real site")
            promote(vs)
        else:
            vs_weights[vs] = weights
    return real_sites, vs_weights


def ring_bonding(real,virtual,A_cg,dihedrals):
    #Constructs constraint structure for ring systems
    
    #Remove all bonds from virtual sites
    for vs in list(virtual.keys()):
        for i in range(A_cg.shape[0]):
            A_cg[vs,i] = 0
            A_cg[i,vs] = 0

    #Construct outer frame
    A_cg[real[0],real[-1]] = 1
    A_cg[real[-1],real[0]] = 1
    for r in range(len(real)-1):
        A_cg[real[r],real[r+1]] = 1
        A_cg[real[r+1],real[r]] = 1
    
    #Construct inner frame and hinge dihedrals
    n_struts = len(real)-3
    j = len(real)-1
    k = 1
    struts = 0
    for s in range(int(math.ceil(n_struts/2.0))):
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = (j+1)%len(real) #First one loops round to 0
        l = k+1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        k += 1
        if struts == n_struts:
            break
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = k-1
        l = j-1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        j -= 1

    return A_cg,dihedrals
        

def get_masses(bead_types,virtual_sites):
    # Get masses, including setting virtual sites to 0.
    # Regular beads (no T/S prefix) can be virtual sites too.
    masses = []
    for b, bead in enumerate(bead_types):
        if b in virtual_sites:
            masses.append(MARTINI_VS_MASS)
        elif bead[0] == 'T':
            masses.append(MARTINI_T_MASS)
        elif bead[0] == 'S':
            masses.append(MARTINI_S_MASS)
        else:
            masses.append(MARTINI_MASS)

    return masses
            

def write_itp(mol, mol_name, beads, bead_types, ring_beads, coords0, charges, A_cg, itp_name):
    # writes gromacs topology file
    nconfs = mol.GetNumConformers()
    with open(itp_name,'w') as itp:
        itp.write('[moleculetype]\n')
        itp.write('{}    2\n'.format(mol_name))
        virtual, real, bead_names = write_atoms(itp,A_cg,mol_name,bead_types,charges,coords0,ring_beads)
        bonds,constraints,dihedrals = write_bonds(mol, beads, nconfs, itp, A_cg, ring_beads, real,virtual)
        angles = write_angles(mol, beads, nconfs, itp, bonds, constraints)
        if dihedrals:
            write_dihedrals(itp,dihedrals,coords0)
        if virtual:
            write_virtual_sites(itp, virtual, beads)
    return bead_names
            

def write_atoms(itp,A_cg,mol_name,bead_types,charges,coords,ring_beads):
    # Writes [atoms] block in itp file
    real = []
    virtual = {}
    # Split ring beads into real and virtual sites
    for ring in ring_beads:
        rs,vs = get_virtual_sites(ring,coords,A_cg)
        virtual.update(vs)
        real.append(rs)

    masses = get_masses(bead_types,virtual)

    itp.write('\n[atoms]\n')
    bead_names = []
    for b in range(len(bead_types)):
        if masses[b] == 0:
            bead_name = 'V' + str(b+1)
        else:
            bead_name = 'B' + str(b+1)
        bead_names.append(bead_name)
        itp.write('{:5d}{:>5}{:5d}{:>5}{:>5}{:5d}{:>10.3f}{:>10.3f};\n'.format(b+1,bead_types[b],1,mol_name,'B'+str(b+1),b+1,charges[b],masses[b]))

    return virtual, real, bead_names

def non_aromatic_ring_bead_indices(mol: Chem.Mol, beads: list[int]):
    """
    Return sorted atom indices that are in at least one non-aromatic ring.
    Ring aromaticity is decided by: all bonds in that ring are aromatic -> aromatic ring.
    Uses SSSR rings from mol.GetRingInfo().AtomRings().
    """
    # Make sure aromaticity/rings are perceived
    # (safe if already sanitized)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    ri = mol.GetRingInfo()
    non_arom_atoms = set()

    for ring in ri.AtomRings():  # each ring is a tuple of atom indices
        ring = list(ring)
        # collect bond indices along the ring
        bond_ids = []
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            b = mol.GetBondBetweenAtoms(a1, a2)
            if b is None:
                # Shouldn't happen for valid rings, but guard anyway
                bond_ids = None
                break
            bond_ids.append(b.GetIdx())

        if bond_ids is None:
            continue

        ring_is_aromatic = all(mol.GetBondWithIdx(bi).GetIsAromatic() for bi in bond_ids)

        if not ring_is_aromatic:
            non_arom_atoms.update(ring)

    non_arom_beads = [bi for bi, bead in enumerate(beads) if non_arom_atoms.intersection(bead)]
    return sorted(non_arom_beads)

def write_bonds(mol, beads, nconfs, itp,A_cg,ring_atoms,real,virtual):
    # Writes [bonds] and [constraints] blocks in itp file
    # Construct bonded structures for ring systems, including dihedrals
    if nconfs < 1:
        raise RuntimeError("no conformers are available for bond lengths")
    dihedrals = []
    for r,ring in enumerate(ring_atoms):
        A_cg,dihedrals = ring_bonding(real[r],virtual,A_cg,dihedrals)
    
    itp.write('\n[bonds]\n')
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = MARTINI_BOND_K

    # Get average bond lengths from all conformers
    rs = np.zeros(len(bonds))
    coords = np.zeros((len(beads),3))
    for conf in mol.GetConformers():
        for i,bead in enumerate(beads):
            coords[i] = bead_coords(bead, conf, mol)
        for b,bond in enumerate(bonds):
            rs[b] += np.linalg.norm(np.subtract(coords[bond[0]],coords[bond[1]]))/nconfs
    
    # create a list of beads in non-aromatic rings
    non_am_ring_beads = set(non_aromatic_ring_bead_indices(mol, beads))
    
    # remove dihedrals in non-aromatic rings:   
    dihedrals = [d for d in dihedrals if all(x not in non_am_ring_beads for x in d)]
            

    # Split into bonds and constraints, and write bonds
    con_rs = []
    for bond,r in zip(bonds,rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                if bond[0] not in non_am_ring_beads and bond[1] not in non_am_ring_beads:
                    share_ring = True
                    constraints.append(bond)
                    con_rs.append(r)
                    break
        if not share_ring:
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0]+1,bond[1]+1,1,r,k))

    # Write constraints
    if len(constraints) > 0:
        itp.write('\n#ifdef min\n')
        k = MARTINI_RING_MIN_K #Very stiff bond for minimisation of rings
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(con[0]+1,con[1]+1,1,r,k))

        itp.write('\n#else\n')
        itp.write('[constraints]\n')
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}\n'.format(con[0]+1,con[1]+1,1,r))
        itp.write('#endif\n')

    return bonds,constraints,dihedrals

def write_angles(mol, beads, nconfs, itp, bonds, constraints):
    # Writes [angles] block in itp file
    k = MARTINI_ANGLE_K

    # Get list of angles
    angles = []
    for bi in range(len(bonds)-1):
        for bj in range(bi+1,len(bonds)):
            shared = np.intersect1d(bonds[bi],bonds[bj])
            if np.size(shared) == 1:
                if bonds[bi] not in constraints or bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x,int(shared[0]),z])
    # Calculate and write to file
    if angles:
        itp.write('\n[angles]\n')
        coords = np.zeros((len(beads),3), dtype=float)
        thetas = np.zeros(len(angles), dtype=float)
        for conf in mol.GetConformers():
            for i,bead in enumerate(beads):
                coords[i] = bead_coords(bead, conf, mol)
            for a,angle in enumerate(angles):
                vec1 = np.subtract(coords[angle[0]],coords[angle[1]])
                vec1 = vec1/np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]],coords[angle[1]])
                vec2 = vec2/np.linalg.norm(vec2)
                theta = np.arccos(np.dot(vec1,vec2))
                thetas[a] += theta
                #print(vec1,vec2)

        thetas = thetas*180.0/(np.pi*nconfs)


        for a,t in zip(angles,thetas):
            itp.write('{:5d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(a[0]+1,a[1]+1,a[2]+1,2,t,k))


def _signed_dihedral_deg(coords, indices):
    """Signed dihedral i-j-k-l in degrees, from atan2 of the bond normals.

    The sign follows the right-hand rule about j-k. arccos of the normal
    dot product would fold that sign away and report only 0 to 180.
    """
    points = [np.asarray(coords[idx], dtype=float) for idx in indices]
    bond_ij = points[1] - points[0]
    bond_jk = points[2] - points[1]
    bond_kl = points[3] - points[2]
    normal_1 = np.cross(bond_ij, bond_jk)
    normal_2 = np.cross(bond_jk, bond_kl)
    axis = bond_jk / np.linalg.norm(bond_jk)
    return float(np.degrees(np.arctan2(
        np.dot(np.cross(normal_1, axis), normal_2),
        np.dot(normal_1, normal_2),
    )))


def write_dihedrals(itp,dihedrals,coords0):
    # Hinge dihedrals chosen in ring_bonding, from the first conformer.
    itp.write('\n[dihedrals]\n')
    k = MARTINI_DIH_K

    for dih in dihedrals:
        angle = _signed_dihedral_deg(coords0, dih)
        itp.write('{:5d}{:3d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(dih[0]+1,dih[1]+1,dih[2]+1,dih[3]+1,2,angle,k))

def write_virtual_sites(itp,virtual_sites, beads):
    # Write [virtual_sites] block to itp file
    itp.write('\n[virtual_sitesn]\n')
    
    vs_iter = sorted(virtual_sites.keys())

    for vs in vs_iter:
        cs = sorted(virtual_sites[vs].items())
        if len(cs) == 4:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1],cs[3][0]+1,cs[3][1]))
        elif len(cs) == 3:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1]))
    
    itp.write('\n[exclusions]\n')
    
    done = []

    #Add exclusions between vs and all other beads
    for vs in vs_iter:
        excl = str(vs+1)
        for i in range(len(beads)):
            if i != vs and i not in done:
                excl += ' '+str(i+1)
        done.append(vs)
        itp.write('{}\n'.format(excl))

def get_coords(mol, beads):
    """Heavy-atom center of geometry for every bead, in nm."""
    if mol.GetNumConformers() < 1:
        raise RuntimeError("molecule has no conformer; coordinates cannot be written")
    conf = mol.GetConformer(0)
    return np.array([bead_coords(bead, conf, mol) for bead in beads])
