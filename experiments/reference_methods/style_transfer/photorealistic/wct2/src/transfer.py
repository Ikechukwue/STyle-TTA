"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
from experiments.reference_methods.style_transfer.photorealistic.wct2.src.model import WaveEncoder, WaveDecoder
from experiments.reference_methods.style_transfer.photorealistic.wct2.src.utils.core import feature_wct
from experiments.reference_methods.style_transfer.photorealistic.wct2.src.utils.io import Timer, compute_label_info


class WCT2:
    def __init__(self, weights, transfer_at=['encoder', 'decoder', 'skip'], option_unpool='cat5', device='cuda', verbose=False):
        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'
        self.option_unpool = option_unpool
        self.verbose = verbose
        self.device = device

        self.encoder = WaveEncoder(option_unpool).to(device)
        self.decoder = WaveDecoder(option_unpool).to(device)
        
        if weights:
            self.encoder.load_state_dict(weights['encoder'])
            self.decoder.load_state_dict(weights['decoder'])
            
        self.encoder.eval()
        self.decoder.eval()

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        with torch.no_grad():
            # 1. Encode content level-by-level, applying WCT at each encoder level
            with Timer("Elapsed time in encoding: %f"):
                for level in [1, 2, 3, 4]:
                    content_feat = self.encode(content_feat, content_skips, level)
                    if 'encoder' in self.transfer_at and level in wct2_enc_level:
                        content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                                   content_segment, style_segment,
                                                   label_set, label_indicator,
                                                   alpha=alpha, device=self.device)
                        self.print_('transfer at encoder {}'.format(level))

            # 2. Transfer at skip connections
            with Timer("Elapsed time in transfer: %f"):
                if 'skip' in self.transfer_at:
                    for skip_level in wct2_skip_level:
                        for component in [0, 1, 2]:  # component: [LH, HL, HH]
                            content_skips[skip_level][component] = feature_wct(
                                content_skips[skip_level][component], style_skips[skip_level][component],
                                content_segment, style_segment,
                                label_set, label_indicator,
                                alpha=alpha, device=self.device)
                        self.print_('transfer at skip {}'.format(skip_level))

            # 3. Decode, applying WCT at each decoder level
            with Timer("Elapsed time in decoding: %f"):
                for level in [4, 3, 2, 1]:
                    if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                        content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                                   content_segment, style_segment,
                                                   label_set, label_indicator,
                                                   alpha=alpha, device=self.device)
                        self.print_('transfer at decoder {}'.format(level))
                    content_feat = self.decode(content_feat, content_skips, level)
            
            return content_feat
