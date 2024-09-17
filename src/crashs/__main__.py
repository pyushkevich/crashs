import argparse
from crashs.crashs import FitLauncher
from crashs.build_template import BuildTemplateLauncher
from crashs.roi_integrate import ROILauncher
from crashs.profile_sampling import ProfileSamplingLauncher, ProfileMappingLauncher

# Create a parser
parse = argparse.ArgumentParser(
    prog="crashs", description="CRASHS: Cortical reconstruction for ASHS")

# Add subparsers for the main commands
sub = parse.add_subparsers(dest='command', help='sub-command help', required=True)

# Add the CRASHS subparser commands
c_fit = FitLauncher(
    sub.add_parser('fit', help='Fit ASHS segmentation to CRASHS template'))

c_build = BuildTemplateLauncher(
    sub.add_parser('build', help='Build new CRASHS template'))

c_roi = ROILauncher(
    sub.add_parser('roi', help='Integrate features over ROIs in a mesh'))

c_prof_sam = ProfileSamplingLauncher(
    sub.add_parser('profile_sample', help='Sample data into template space using CRASHS profiles'))
    
c_prof_map = ProfileMappingLauncher(
    sub.add_parser('profile_map', help='Map data from template to subject using CRASHS profiles'))

# Parse the arguments
args = parse.parse_args()
args.func(args)
