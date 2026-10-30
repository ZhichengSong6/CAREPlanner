"""Shared physical policy for a known plane and bounded observation supports."""
from pathlib import Path
import hashlib,json,math
from numbers import Integral
import xml.etree.ElementTree as ET

def load_ground_policy(urdf,geometry):
    contract=json.loads(Path(geometry).read_text())
    if contract.get('schema')==1:
        if contract['margin_m']!=.02:raise ValueError('invalid_legacy_ground_margin')
        return contract,[.02]*len(contract['bodies'])
    research=contract.get('schema')==4 and contract.get('research_profile')=='nominal_plane_point_v1'
    tracking_budget=0. if research else .02
    if contract.get('schema') not in (2,3,4) or (contract.get('schema')==4 and not research) or contract.get('physical_margin_m')!=.02 or contract.get('assumed_tracking_tube_m')!=tracking_budget or contract.get('margin_m')!=.02+tracking_budget:
        raise ValueError('invalid_ground_support_policy')
    uncertainty=contract.get('plane_uncertainty_m')
    if not isinstance(uncertainty,(int,float)) or not math.isfinite(uncertainty) or uncertainty<0:
        raise ValueError('invalid_ground_uncertainty')
    if contract['schema'] in (3,4):
        plane=contract['plane'];normal=plane['normal'];offset=plane['offset_m']
        if (plane['frame']!='base_link' or plane['support_id']!='known_ground' or
            not isinstance(plane['calibration_id'],str) or not plane['calibration_id'] or
            len(normal)!=3 or not all(math.isfinite(x) for x in normal) or
            abs(sum(x*x for x in normal)-1.)>1e-10 or not math.isfinite(offset)):
            raise ValueError('invalid_analytic_plane')
        observation_reserve=0.
        if plane.get('association_enabled',False):
            registration=plane.get('registration_bound_m');band=plane.get('association_band_m',.002)
            if (not isinstance(registration,(int,float)) or isinstance(registration,bool) or not math.isfinite(registration) or registration<0 or
                not isinstance(band,(int,float)) or isinstance(band,bool) or not math.isfinite(band) or band<0 or not math.isfinite(registration+band)):
                raise ValueError('invalid_ground_observation_bound')
            observation_reserve=0. if research else registration+band
    elif not math.isfinite(contract['plane_z_m']) or uncertainty!=0.:
        raise ValueError('unsupported_legacy_ground_prior')
    else:observation_reserve=0.
    if contract.get('contact_numerical_tolerance_m')!=1e-8:raise ValueError('invalid_contact_numerics')
    robot=ET.parse(urdf).getroot()
    required={(link.get('name'),i) for link in robot.findall('link') for i,_ in enumerate(link.findall('collision'))}
    entries=[(b['link'],b['collision_index']) for b in contract['bodies']]
    if len(entries)!=len(set(entries)) or set(entries)!=required:raise ValueError('incomplete_ground_collision_geometry')
    root=Path(urdf).resolve().parents[3]
    for name,digest in contract['sources'].items():
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('ground_geometry_source_changed')
    parents={j.find('child').get('link'):(j.find('parent').get('link'),j.get('type')) for j in robot.findall('joint')}
    floors=[]
    for body in contract['bodies']:
        name=body['link'];moving=False
        while name!='base_link':
            name,kind=parents[name];moving=moving or kind!='fixed'
        mounting=body['contact_policy']=='mounted_nonpenetration'
        if mounting and (body['link']!='base_link' or moving):raise ValueError('invalid_ground_mount_contact')
        physical=0. if mounting else .02;tracking=tracking_budget if moving else 0.
        if (body['support_id']!='known_ground' or type(body['structurally_fixed']) is not bool or body['structurally_fixed']!=(not moving) or
            body['physical_margin_m']!=physical or body['assumed_tracking_normal_reserve_m']!=tracking or
            not math.isfinite(body['reference_margin_m']) or abs(body['reference_margin_m']-physical-tracking)>1e-12):raise ValueError('invalid_ground_body_margin')
        floors.append(physical+tracking+uncertainty+(0. if mounting else observation_reserve)-(1e-8 if mounting else 0.))
    return contract,floors

def classify_ground_support(hit_count,invalid_count,z_min,z_max,registration_bound_m=None,association_band_m=.002):
    """No association is permitted without an independently established bound.

    This returns a conservative support upper height, never a FREE update.
    An enabled caller must retain every other/unknown component of a mixed
    voxel and bind the original ray times and source geometry before use.
    """
    if registration_bound_m is None:return dict(eligible=False,reason='registration_bound_unestablished')
    if not math.isfinite(registration_bound_m) or registration_bound_m<0:return dict(eligible=False,reason='invalid_registration_bound')
    if not math.isfinite(association_band_m) or association_band_m<0:return dict(eligible=False,reason='invalid_association_band')
    if any(not isinstance(v,Integral) or isinstance(v,bool) or v<0 for v in (hit_count,invalid_count)):
        return dict(eligible=False,reason='invalid_source_counts')
    if hit_count<=0 or invalid_count:return dict(eligible=False,reason='incomplete_source_provenance')
    if not all(math.isfinite(v) for v in (z_min,z_max)) or z_min>z_max:return dict(eligible=False,reason='invalid_support_range')
    if z_min < -association_band_m or z_max > association_band_m:return dict(eligible=False,reason='mixed_or_non_ground_support')
    return dict(eligible=True,reason='conditional_bounded_support',upper_z_m=z_max+registration_bound_m,
        ground_only_proven=False,changes_occupancy=False)
