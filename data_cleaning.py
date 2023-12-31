import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# salary parsing
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))
df['min_salary'] = min_hr.apply(lambda x: x.split('-')[0])
df['min_salary'] = df['min_salary'].astype('int')
df['max_salary'] = min_hr.apply(lambda x: x.split('-')[1])
df['max_salary'] = df['max_salary'].astype('int')
df['avg_salary'] = (df.min_salary+df.max_salary)/2

# company name
# df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'].split('\n')[0], axis = 1)
# df['company_txt'] = df.apply(lambda x: x['Company Name'].split('\n')[0], axis = 1)
df['company_txt'] = df['Company Name'].apply(lambda x: x.split('\n')[0])

# state field
df['job_state'] = df['Location'].apply(lambda x: x.split(', ')[-1])
# df['job_state'] = df['Location'].apply(lambda x: 'CA' if x.job_state == 'Los Angeles' else x.job_state)
print(df.job_state.value_counts())

# same loc as headquarters
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# age of company
df['age'] = df.Founded.apply(lambda x: x if x<1 else 2023 - x)

# parsing job description
# looking for python, rstudio, spark, aws, excel
df['python_yn']=df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['rstudio_yn']=df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() or 'rstudio' in x.lower() else 0)
df['spark_yn']=df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['aws_yn']=df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['excel_yn']=df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
print(df.python_yn.value_counts())
print(df.rstudio_yn.value_counts())
print(df.spark_yn.value_counts())
print(df.aws_yn.value_counts())
print(df.excel_yn.value_counts())

print(df.columns)
df.out = df.drop(['Unnamed: 0'], axis = 1)

df.out.to_csv('file.csv', index=False)