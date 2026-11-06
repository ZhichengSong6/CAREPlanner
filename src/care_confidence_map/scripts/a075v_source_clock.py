"""Explicit A075V device-clock mapping; no receive-time replacement.

The official SDK documents uint64 device milliseconds. A millisecond counter
does not establish a ROS epoch or the exposure phase. Those are input contracts.
"""
from dataclasses import dataclass
from numbers import Integral

def integer(value,name):
    if not isinstance(value,Integral) or isinstance(value,bool) or value<0:raise ValueError('invalid_'+name)
    return int(value)

def device_milliseconds_parts(device_ms):
    seconds,remainder=divmod(integer(device_ms,'device_ms'),1000)
    return seconds,remainder*1000000

@dataclass(frozen=True)
class SourceClockModel:
    calibration_id:str
    boot_id:str
    device_reference_ms:int
    ros_reference_ns:int
    offset_uncertainty_ns:int
    phase_uncertainty_ns:int
    rate_error_ppm:int
    validity_ms:int

    def map_time(self,device_ms):
        for name in ['device_reference_ms','ros_reference_ns','offset_uncertainty_ns','phase_uncertainty_ns','rate_error_ppm','validity_ms']:
            integer(getattr(self,name),name)
        if not self.calibration_id or not self.boot_id or not self.validity_ms:raise ValueError('missing_clock_identity_or_validity')
        delta=integer(device_ms,'device_ms')-self.device_reference_ms
        if abs(delta)>self.validity_ms:raise ValueError('clock_model_outside_validity')
        # delta_ms*1e6ns times ppm/1e6 = abs(delta_ms)*ppm ns.
        uncertainty=self.offset_uncertainty_ns+self.phase_uncertainty_ns+1000000+abs(delta)*self.rate_error_ppm
        mapped=self.ros_reference_ns+delta*1000000
        if mapped<=0:raise ValueError('nonpositive_ros_source_stamp')
        return mapped,uncertainty

class A075VSourceClock:
    def __init__(self,model=None):self.model=model;self.last_frame=None;self.last_device_ms=None;self.reset_fault=False
    def observe(self,frame_id,device_ms,receive_ros_ns,boot_id,maximum_age_ns=50000000):
        frame_id=integer(frame_id,'frame_id');device_ms=integer(device_ms,'device_ms');receive_ros_ns=integer(receive_ros_ns,'receive_ros_ns')
        base=dict(frame_id=str(frame_id),device_stamp_ms=str(device_ms),receive_ros_ns=str(receive_ros_ns),source_time_valid=False)
        if self.reset_fault:return dict(base,reason='device_epoch_recalibration_required')
        if self.last_frame is not None and (frame_id<self.last_frame or device_ms<self.last_device_ms):
            self.reset_fault=True
            return dict(base,reason='reordered_or_device_reset')
        if self.last_frame is not None and (frame_id==self.last_frame or device_ms==self.last_device_ms):
            return dict(base,reason='duplicate_frame_or_stamp')
        self.last_frame=frame_id;self.last_device_ms=device_ms
        if self.model is None:return dict(base,reason='device_ros_clock_unestablished')
        if boot_id!=self.model.boot_id:return dict(base,reason='device_boot_epoch_changed')
        try:stamp,uncertainty=self.model.map_time(device_ms)
        except ValueError as exc:return dict(base,reason=str(exc))
        # The entire acquisition-time interval must be before receive time,
        # positive and fresh. No estimate is silently clamped to ROS now.
        if stamp-uncertainty<=0 or stamp+uncertainty>receive_ros_ns:
            return dict(base,reason='source_interval_future_or_nonpositive')
        if receive_ros_ns-(stamp-uncertainty)>integer(maximum_age_ns,'maximum_age_ns'):
            return dict(base,reason='source_interval_stale')
        seconds,nanoseconds=divmod(stamp,1000000000)
        if seconds>0xffffffff:return dict(base,reason='ros1_time_range_exceeded')
        return dict(base,source_time_valid=True,reason='calibrated_device_clock_interval',source_sec=seconds,source_nsec=nanoseconds,
            source_time_uncertainty_ns=str(uncertainty),calibration_id=self.model.calibration_id,boot_id=boot_id)
