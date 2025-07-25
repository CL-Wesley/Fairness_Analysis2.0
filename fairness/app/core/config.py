

FILE_MODEL_DOWNLOAD_API = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download"
# FILE_MODEL_DOWNLOAD_API = "http://localhost:8000/api/files_download"
ALLOWED_ANALYSIS_TYPES = {
    "Data Drift": "datadrift",
    "Fairness":"fairness",
    "Classification Stats":"classification"

}

def check_analysis(analysis_type):

    analysis = ""
    for key,val in ALLOWED_ANALYSIS_TYPES.items():
        if analysis_type in key:
            analysis=val
        else:
            continue
    return analysis