import requests

############################################################

fetch_jobs_url = 'http://china-gpu02.sigsus.cn/ai_arts/api/trainings/?pageNum=1&pageSize=100000&status=all&vcName=platform'

############################################################


payloadHeader = {
    'Host': 'china-gpu02.sigsus.cn',
    'Content-Type': 'application/json',
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjMwMDAxLCJ1c2VyTmFtZSI6InlpcmFuLnd1IiwiZXhwIjoxNjU3MzI3OTYzLCJpYXQiOjE2MjEzMjc5NjN9.qvf2_JvSnpeReDMSGkvpaMX0dPRobCcDdKHIkyzsLtw"
}

res = requests.get(fetch_jobs_url, headers=payloadHeader).json()
jobs = res["data"]["Trainings"]
for job in jobs:
    job_id = job["id"]
    status = job["status"]
    if status == "queued":
        stop_job_url = 'http://china-gpu02.sigsus.cn/ai_arts/api/trainings/' + job_id
        res = requests.delete(stop_job_url, headers=payloadHeader).json()
        delete_job_url = 'http://china-gpu02.sigsus.cn/ai_arts/api/inferences/DeleteJob?jobId=' + job_id
        res = requests.delete(delete_job_url, headers=payloadHeader).json()
        print(res)
