"""
Original code:
    https://github.com/Trusted-AI/AIF360
"""
import os

import pandas as pd

from data_handler.AIF360.standard_dataset import StandardDataset


class CreditDataset(StandardDataset):
    """Bank marketing Dataset.
    See :file:`aif360/data/raw/bank/README.md`.
    """

    def __init__(self, root_dir='./data/credit', label_name='default payment next month', favorable_classes=[1],
                 protected_attribute_names=['SEX'],
                 privileged_classes=[lambda x: x == 1],
                 instance_weights_name=None,
                 categorical_features=['EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                                       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.
        By default, this code converts the 'marital' attribute to a binary value
        where privileged is `not married` and unprivileged is `married` as in
        :obj:`GermanDataset`.
        """

        filepath = os.path.join(root_dir, 'credit.csv')

        try:
            df = pd.read_csv(filepath)
        except IOError as err:
            print("IOError: {}".format(err))
            import sys
            sys.exit(1)

        super(CreditDataset, self).__init__(df=df, label_name=label_name,
                                            favorable_classes=favorable_classes,
                                            protected_attribute_names=protected_attribute_names,
                                            privileged_classes=privileged_classes,
                                            instance_weights_name=instance_weights_name,
                                            categorical_features=categorical_features,
                                            features_to_keep=features_to_keep,
                                            features_to_drop=features_to_drop, na_values=na_values,
                                            custom_preprocessing=custom_preprocessing, metadata=metadata)
