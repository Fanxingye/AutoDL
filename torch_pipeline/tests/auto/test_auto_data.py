# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test auto data set"""
import os
import tempfile
import pandas as pd
from autotorch.auto.data import TorchImageClassificationDataset


def test_image_datasets_from_csv():
    N = 10
    def gen_fake_data(dir_name):
        paths = [f'{i}.jpg' for i in range(N)]
        labels = [i for i in range(N)]
        df = pd.DataFrame({'image': paths, 'label': labels})
        df.to_csv(os.path.join(dir_name, 'normal.csv'), index=False)
        df1 = pd.DataFrame({'some_image': paths, 'some_label': labels})
        df1.to_csv(os.path.join(dir_name, 'abnormal.csv'), index=False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        gen_fake_data(tmpdirname)
        normal_csv = os.path.join(tmpdirname, 'normal.csv')
        abnormal_csv = os.path.join(tmpdirname, 'abnormal.csv')
        d1 = TorchImageClassificationDataset.from_csv(normal_csv, root=tmpdirname)
        assert 'image' in d1.columns
        assert 'label' in d1.columns
        d1 = TorchImageClassificationDataset(normal_csv)
        d2 = TorchImageClassificationDataset(abnormal_csv, image_column='some_image', label_column='some_label')
        d2 = TorchImageClassificationDataset.from_csv(abnormal_csv, image_column='some_image', label_column='some_label')


if __name__ == '__main__':
    import nose
    nose.runmodule()
