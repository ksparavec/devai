-- job_submit.lua -- devai-slurm accepts jobs from the Slurm user devai only
-- (docs/slurm.md Sec. 3.1 and 12; platform-rebuild C3, I6).
--
-- The HS256 key (jwt_hs256.key) could sign a token for ANY user, root
-- included. This filter refuses every job whose submitter or owner is not
-- uid 1000 (devai), so a key holder gets at most a devai job, which runs as a
-- subordinate uid on the host (I26). Installed as /etc/slurm/job_submit.lua
-- (JobSubmitPlugins=lua; Debian's job_submit_lua.so).

local DEVAI_UID = 1000

function slurm_job_submit(job_desc, part_list, submit_uid)
	if submit_uid ~= DEVAI_UID then
		slurm.log_user("devai jobs run as devai only")
		slurm.log_info("job_submit.lua: refused a job submitted by uid %s", tostring(submit_uid))
		return slurm.ERROR
	end
	if job_desc.user_id ~= nil and job_desc.user_id ~= DEVAI_UID then
		slurm.log_user("devai jobs run as devai only")
		slurm.log_info("job_submit.lua: refused a job for uid %s", tostring(job_desc.user_id))
		return slurm.ERROR
	end
	return slurm.SUCCESS
end

-- Every change passes: root's scontrol, the epilog's AdminComment. A job's
-- user cannot be changed in Slurm.
function slurm_job_modify(job_desc, job_rec, part_list, modify_uid)
	return slurm.SUCCESS
end

return slurm.SUCCESS
