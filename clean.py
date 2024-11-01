def clean_requirements():
    with open('requirements.txt', 'r') as file:
        requirements = file.readlines()

    cleaned_requirements = []
    for req in requirements:
        # Skip lines with @ symbols (local file paths and git repos)
        if '@' in req:
            # Extract just the package name for git repositories
            if 'git+' in req:
                package_name = req.split('#')[0].split('/')[-1].split('.git')[0]
                cleaned_requirements.append(f"{package_name}\n")
            continue
        # Skip relative path installations (-e)
        if req.startswith('-e'):
            continue
        # Skip empty lines
        if not req.strip():
            continue
        cleaned_requirements.append(req)

    # Write cleaned requirements back to file
    with open('requirements.txt', 'w') as file:
        file.writelines(sorted(cleaned_requirements))

clean_requirements()