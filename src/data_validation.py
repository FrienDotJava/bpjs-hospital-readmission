import pandas as pd
import great_expectations as gx
from utils import load_params, load_dataset_from_csv
from pathlib import Path

class BPJSValidator:
    def __init__(self, datasource_name: str = "bpjs_datasource"):
        self.context = gx.get_context(mode="file")
        self.data_source = self.context.data_sources.add_or_update_pandas(datasource_name)

    
    def validate_dataset(self, df: pd.DataFrame, name: str):
        suite_name = f"{name}_validation"
        asset_name = f"{name}_asset"
        cp_name = f"checkpoint_{name}"

        asset = self.data_source.add_dataframe_asset(asset_name)
        batch_definition = asset.add_batch_definition_whole_dataframe(name=f"{name}_batch_definition")

        suite = self.context.suites.get(suite_name)

        checkpoint = self.context.checkpoints.add_or_update(
            gx.Checkpoint(
                name=cp_name,
                validation_definitions=[
                    gx.ValidationDefinition(
                        name=f"{name}_audit",
                        data=batch_definition,
                        suite=suite,
                    ),
                ],
            )
        )

        return checkpoint.run(batch_parameters={"dataframe": df})


def main():
    validator = BPJSValidator()

    datasets = ['peserta', 'fktp', 'fkrtl']
    params = load_params()

    for dataset_name in datasets:
        path = Path(params['data']['raw_data_folder']) / f"{dataset_name}.csv"
        df = load_dataset_from_csv(path)

        print(f"Validating {dataset_name} dataset")
        result = validator.validate_dataset(df, dataset_name)

        if not result.success:
            print(f"Validation failed for {dataset_name}. Check Data Docs.")
        else:
            print(f"{dataset_name} passed validation.")

    

if __name__ == "__main__":
    main()