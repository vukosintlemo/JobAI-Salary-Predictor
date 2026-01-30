import pandas as pd
import os


class JobDataLoader:
    def __init__(self):
        # 1. Get the absolute path of the directory where data_loader.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Go up one level to the project root (JobAI), then into the 'data' folder
        self.data_dir = os.path.join(current_dir, "..", "data")

    def load_and_clean(self):
        # Construct full paths to files
        jobs_path = os.path.join(self.data_dir, "ai_jobs.csv")
        skills_path = os.path.join(self.data_dir, "skills_demand.csv")
        mapping_path = os.path.join(self.data_dir, "job_title_mapping.csv")

        # Check if the file actually exists before trying to read it
        if not os.path.exists(jobs_path):
            raise FileNotFoundError(
                f"Could not find the file at: {os.path.abspath(jobs_path)}"
            )
        # 1. Load the raw files
        jobs = pd.read_csv(jobs_path)
        skills = pd.read_csv(skills_path)
        mapping = pd.read_csv(mapping_path)

        # 2. Aggregation: Create the 'skill_count' feature
        # This counts how many skills are required for each job_id
        skill_counts = skills.groupby("job_id").size().reset_index(name="skill_count")

        # 3. Merging: Bring it all together
        # Join jobs with skill counts (on job_id)
        df = jobs.merge(skill_counts, on="job_id", how="left")

        # 4. Fill Missing Values
        # If a job wasn't in the skills file, it has 0 skilld (not NaN)
        df["skill_count"] = df["skill_count"].fillna(0)

        # 5. Create the Target variable for ML
        df["salary_median_usd"] = (df["salary_min_usd"] + df["salary_max_usd"]) / 2

        return df


if __name__ == "__main__":
    # Test the Loader
    loader = JobDataLoader()
    data = loader.load_and_clean()
    print(f"Load {len(data)} jobs with column: {data.columns.tolist()}")
