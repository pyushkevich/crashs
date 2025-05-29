# The main program launcher
from huggingface_hub import snapshot_download, configure_http_backend
import argparse
import requests

def _make_session(https_verify: bool) -> requests.Session:
    """
    Create a requests session with the specified HTTPS verification setting.
    """
    session = requests.Session()
    session.verify = https_verify
    return session

class HuggingFaceImporter:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('crashs_data', metavar='crashs_data', type=str, help='Path to the CRASHS data directory where files will be downloaded')
        parse.add_argument("-k", "--insecure", action="store_true", help="Skip HTTPS certificate verification")
        parse.add_argument("-r", "--repo", type=str, default=None, help="Alternative Hugging Face repository to download from")
    
        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        # Load the template
        configure_http_backend(backend_factory=lambda: _make_session(not args.insecure))
        repository = args.repo if args.repo else "pyushkevich/crashs_template_package"
        local_path = snapshot_download(repo_id=repository, local_dir=args.crashs_data, repo_type="dataset")
        print(f"CRASHS template package downloaded to {local_path}")

