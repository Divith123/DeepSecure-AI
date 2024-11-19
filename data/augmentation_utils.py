import cv2
import librosa
import numpy as np
import albumentations
from albumentations import (Compose, ImageCompression, GaussNoise, HorizontalFlip, 
                            PadIfNeeded, OneOf,ToGray, ShiftScaleRotate, GaussianBlur,
                            RandomBrightnessContrast, FancyPCA, HueSaturationValue, BasicTransform)


class AudioTransform(BasicTransform):
    """ Transform for audio task. This is the main class where we override the targets and update params function for our need"""
    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

class TimeShifting(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(TimeShifting, self).__init__(always_apply, p)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        start_ = int(np.random.uniform(-80000,80000))
        if start_ >= 0:
            audio_time_shift = np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)]
        else:
            audio_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]]
        
        return audio_time_shift

class PitchShift(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5 , n_steps=None):
        super(PitchShift, self).__init__(always_apply, p)
        '''
        nsteps here is equal to number of semitones
        '''
        
        self.n_steps = n_steps
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        return librosa.effects.pitch_shift(data,sr=16000,n_steps=self.n_steps)


class AddGaussianNoise(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(AddGaussianNoise, self).__init__(always_apply, p)
        
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        ''' 
        noise = np.random.randn(len(data))
        data_wn = data + 0.005*noise
        return data_wn


create_frame_transforms = Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),])



create_spec_transforms = albumentations.Compose([
        TimeShifting(p=0.9),  # here not p=1.0 because your nets should get some difficulties
        AddGaussianNoise(p=0.8),
        PitchShift(p=0.5,n_steps=4)
    ])
