import pysge

def resume(
    path, batch_opts="-q hep.q", sleep=5, request_resubmission_options=True,
    return_files=False,
):
    return pysge.sge_resume(
        "zinv", path, options=batch_opts, sleep=sleep,
        request_resubmission_options=request_resubmission_options,
        return_files=return_files,
    )
