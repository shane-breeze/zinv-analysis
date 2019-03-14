import awkward as awk
import numpy as np
import numba as nb
from functools import partial

from . import Collection
from zinv.utils.Geometry import DeltaR2, LorTHPMToXYZE, LorXYZEToTHPM
from zinv.utils.NumbaFuncs import get_nth_sorted_object_indices

class GenBosonProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def event(self, event):
        gp = event.GenPart
        flags = gp.statusFlags
        pdgs = gp.pdgId
        status = gp.status
        motheridx = gp.genPartIdxMother

        nb_mask = ((np.abs(pdgs)==11) | (np.abs(pdgs)==12) | (np.abs(pdgs)==13)\
                   | (np.abs(pdgs)==14) | (np.abs(pdgs)==15) | (np.abs(pdgs)==16)\
                   | (np.abs(pdgs)==23) | (np.abs(pdgs)==24)) & (motheridx==0)
        nbosons = np.zeros_like(pdgs.content, dtype=np.int32)
        nbosons[(nb_mask & (np.abs(pdgs)<20)).content] = 1
        nbosons[(nb_mask & (np.abs(pdgs)>=20)).content] = 2
        nbosons = awk.JaggedArray(pdgs.starts, pdgs.stops, nbosons)
        event.nGenBosons = nbosons.sum()/2

        gp_mask = ((flags&1==1) & (flags&(1<<8)==(1<<8)))\
                & (
                    (((np.abs(pdgs)==11) | (np.abs(pdgs)==13)) & (status==1) & ((flags&(1<<2)==0)))\
                    | ((np.abs(pdgs)==15) & (status==2))\
                    | (((np.abs(pdgs)==12) | (np.abs(pdgs)==14) | (np.abs(pdgs)==16)) & (status==1))\
                )

        # Finished with GenPart branches
        genpart_dressedlepidx = genpart_matched_dressedlepton(event, gp_mask)
        event.GenPartBoson = Collection("GenPartBoson", event)
        pt, eta, phi, mass = create_genpart_boson(event, gp_mask, genpart_dressedlepidx)
        event.GenPartBoson_pt = pt
        event.GenPartBoson_eta = eta
        event.GenPartBoson_phi = phi
        event.GenPartBoson_mass = mass

        event.delete_branches([
            "GenPart_genPartIdxMother", "GenPart_status", "GenPart_eta",
            "GenPart_phi", "GenPart_pt", "GenPart_mass",
            "GenDressedLepton_pdgId", "GenDressedLepton_eta",
            "GenDressedLepton_phi", "GenDressedLepton_pt",
            "GenDressedLepton_mass",
        ])

def create_genpart_boson(ev, gp_mask, gdl_idx):
    return create_genpart_boson_jit(
        ev.GenPart.pt[gp_mask].content,
        ev.GenPart.eta[gp_mask].content,
        ev.GenPart.phi[gp_mask].content,
        ev.GenPart.mass[gp_mask].content,
        gdl_idx,
        ev.GenPart.pt[gp_mask].starts,
        ev.GenPart.pt[gp_mask].stops,
        ev.GenDressedLepton.pt.content,
        ev.GenDressedLepton.eta.content,
        ev.GenDressedLepton.phi.content,
        ev.GenDressedLepton.mass.content,
        ev.GenDressedLepton.pt.starts,
        ev.GenDressedLepton.pt.stops,
    )

@nb.njit
def create_genpart_boson_jit(
    gps_pt, gps_eta, gps_phi, gps_mass, gps_gdidx, gps_starts, gps_stops,
    gds_pt, gds_eta, gds_phi, gds_mass, gds_starts, gds_stops,
):

    nev = gps_stops.shape[0]
    pts = np.zeros(nev, dtype=np.float32)
    etas = np.zeros(nev, dtype=np.float32)
    phis = np.zeros(nev, dtype=np.float32)
    masss = np.zeros(nev, dtype=np.float32)

    for iev, (gps_start, gps_stop, gds_start, gds_stop) in enumerate(zip(
        gps_starts, gps_stops, gds_starts, gds_stops,
    )):
        x, y, z, e = 0., 0., 0., 0.
        for igps in range(gps_start, gps_stop):
            igds = gps_gdidx[igps]
            if igds >= 0:
                igds += gds_start
                tx, ty, tz, te = LorTHPMToXYZE(
                    gds_pt[igds], gds_eta[igds], gds_phi[igds], gds_mass[igds],
                )
                x += tx
                y += ty
                z += tz
                e += te
            else:
                tx, ty, tz, te = LorTHPMToXYZE(
                    gps_pt[igps], gps_eta[igps], gps_phi[igps], gps_mass[igps],
                )
                x += tx
                y += ty
                z += tz
                e += te
        t, h, p, m = LorXYZEToTHPM(x, y, z, e)
        pts[iev] = t
        etas[iev] = h
        phis[iev] = p
        masss[iev] = m
    return pts, etas, phis, masss

def genpart_matched_dressedlepton(ev, gpmask):
    return genpart_matched_dressedlepton_jit(
        ev.GenPart.pdgId[gpmask].content,
        ev.GenPart.eta[gpmask].content,
        ev.GenPart.phi[gpmask].content,
        ev.GenPart.pdgId[gpmask].starts,
        ev.GenPart.pdgId[gpmask].stops,
        ev.GenDressedLepton.pdgId.content,
        ev.GenDressedLepton.eta.content,
        ev.GenDressedLepton.phi.content,
        ev.GenDressedLepton.pdgId.starts,
        ev.GenDressedLepton.pdgId.stops,
    )

@nb.njit
def genpart_matched_dressedlepton_jit(
    gps_pdg, gps_eta, gps_phi, gps_starts, gps_stops,
    gds_pdg, gds_eta, gds_phi, gds_starts, gds_stops,
):
    indices = -1*np.ones(gps_pdg.shape[0], dtype=np.int32)
    for iev, (gps_start, gps_stop, gds_start, gds_stop) in enumerate(zip(
        gps_starts, gps_stops, gds_starts, gds_stops,
    )):
        for igps in range(gps_start, gps_stop):
            for igds in range(gds_start, gds_stop):
                matched_pdg = (gps_pdg[igps] == gds_pdg[igds])
                within_dr = (DeltaR2(gps_eta[igps]-gds_eta[igds],
                                     gps_phi[igps]-gds_phi[igds]) < 0.01)
                if matched_pdg and within_dr:
                    indices[igps] = igds - gds_start
                    break
    return indices
