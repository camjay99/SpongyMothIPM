project(
    'SpongyMothIPM'
)

pymod = import('python')
python = pymod.find_installation(pure: false)

if python.language_version().version_compare('< 3.9')
  error('Requires Python >= 3.9')
endif

subdir('SpongyMothIPM')