activate_env() {
    if [ -f ./.*.auto-env ]; then
        for f in $(ls ./.*.auto-env 2>/dev/null); do
            env=$(basename -- "$f")
            env=${env%.auto-env}
	    trimpr="${VIRTUAL_ENV_PROMPT%"${VIRTUAL_ENV_PROMPT##*[![:space:]]}"}"
	    if [ "(${env})" != "${trimpr}" ]; then
		    echo "I WOULDA '(${env})' v. '${trimpr}'"
            fi
        done
    fi
}
activate_env
