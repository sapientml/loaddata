# Copyright 2023 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from sapientml.generator import CodeBlockGenerator
from sapientml.params import Code, Config, Dataset, Task

template_env = Environment(loader=FileSystemLoader(f"{os.path.dirname(__file__)}/templates"), trim_blocks=True)
ROW_THRESHOLD_FOR_SAMPLING = 100000


def _render(tpl, *args, **kwargs):
    code = tpl.render(*args, **kwargs)
    return "\n".join([line for line in code.split("\n") if len(line) > 0]) + "\n\n"


class LoadDataConfig(Config):
    id_columns_for_prediction: Optional[list[str]] = None


class LoadData(CodeBlockGenerator):
    def __init__(self, **kwargs):
        self.config = LoadDataConfig(**kwargs)

    def generate_code(self, dataset: Dataset, task: Task):
        """Generates validation, test, train and predict code snippets for the pipeline.

        This function will update the validation_dataframe, training_dataframe by
        droping the ignore column.

        Parameters
        ----------
        dataset : Dataset
           Object of Dataset class containing the details of dataset.
        task : Task
           Object of Task class containing the details of task.

        Returns
        -------
        dataset, code : Tuple[Dataset, Code]

        """

        code = Code()
        code.validation = code.test = code.train = code.predict = "# *** GENERATED PIPELINE ***\n\n"

        dataset, tmp_code = self._generate_code_load(dataset, task)
        code += tmp_code
        dataset, tmp_code = self._generate_code_concat_train_validation(dataset, task)
        code += tmp_code
        dataset, tmp_code = self._generate_code_split(dataset, task)
        code += tmp_code
        dataset, tmp_code = self._generate_code_subsample(dataset, task)
        code += tmp_code
        # set_index msut be before ignore for the case where ignore_columns contain id_columns_for_prediction
        dataset, tmp_code = self._generate_code_set_index(dataset, task)
        code += tmp_code
        dataset, tmp_code = self._generate_code_ignore(dataset, task)
        code += tmp_code
        dataset, tmp_code = self._generate_code_set_validation_as_test(dataset, task)
        code += tmp_code

        return dataset, code

    def _generate_code_load(self, dataset: Dataset, task: Task):
        code = Code()
        tpl = template_env.get_template("loaddata.py.jinja")
        code.validation += _render(tpl, dataset=dataset, task=task, config=self.config, validation=True)
        code.test += _render(tpl, dataset=dataset, task=task, config=self.config, validation=False)
        tpl = template_env.get_template("loaddata_train.py.jinja")
        code.train += _render(tpl, dataset=dataset, task=task, config=self.config)
        tpl = template_env.get_template("loaddata_predict.py.jinja")
        code.predict += _render(tpl, dataset=dataset, task=task, config=self.config)
        return dataset, code

    def _generate_code_set_index(self, dataset: Dataset, task: Task):
        code = Code()
        id_columns_for_prediction = self.config.id_columns_for_prediction
        if id_columns_for_prediction:
            tpl = template_env.get_template("set_index.py.jinja")
            code.test += _render(tpl, id_columns_for_prediction=id_columns_for_prediction)
            code.predict += _render(tpl, id_columns_for_prediction=id_columns_for_prediction)
        return dataset, code

    def _generate_code_ignore(self, dataset: Dataset, task: Task):
        code = Code()
        ignore_columns = dataset.ignore_columns
        if ignore_columns:
            tpl = template_env.get_template("drop_ignore_columns.py.jinja")
            code.validation += _render(tpl, ignore_columns=ignore_columns, train=True, validation=True, test=False)
            code.test += _render(tpl, ignore_columns=ignore_columns, train=True, validation=False, test=True)
            code.train += _render(tpl, ignore_columns=ignore_columns, train=True, validation=False, test=False)
            code.predict += _render(tpl, ignore_columns=ignore_columns, train=False, validation=False, test=True)
            dataset.training_dataframe = dataset.training_dataframe.drop(
                dataset.ignore_columns, axis=1, errors="ignore"
            )
            if dataset.validation_dataframe is not None:
                dataset.validation_dataframe = dataset.validation_dataframe.drop(
                    dataset.ignore_columns, axis=1, errors="ignore"
                )
            if dataset.test_dataframe is not None:
                dataset.test_dataframe = dataset.test_dataframe.drop(dataset.ignore_columns, axis=1, errors="ignore")
        return dataset, code

    def _generate_code_concat_train_validation(self, dataset: Dataset, task: Task):
        code = Code()
        if dataset.validation_dataframe is not None:
            tpl = template_env.get_template("concat_train_validation.py.jinja")
            code.test += _render(tpl)
        return dataset, code

    def _generate_code_split(self, dataset: Dataset, task: Task):
        code = Code()
        tpl = template_env.get_template("split.py.jinja")
        code.validation += _render(tpl, dataset=dataset, task=task, validation=True)
        code.test += _render(tpl, dataset=dataset, task=task, validation=False)
        return dataset, code

    def _generate_code_subsample(self, dataset: Dataset, task: Task):
        code = Code()
        tpl = template_env.get_template("subsample.py.jinja")
        code.validation += _render(tpl, task=task, sample_size=ROW_THRESHOLD_FOR_SAMPLING)
        return dataset, code

    def _generate_code_set_validation_as_test(self, dataset: Dataset, task: Task):
        code = Code()
        tpl = template_env.get_template("set_validation_as_test.py.jinja")
        code.validation += _render(tpl)
        return dataset, code
