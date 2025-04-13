{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pillow
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.torch
    pkgs.python311Packages.torchvision
    pkgs.python311Packages.scikit-learn
  ];

  shellHook = ''
    # # Create virtual environment
    # python -m venv .venv
    # source .venv/bin/activate
    #
    # # Install remaining packages via pip
    # pip install --upgrade pip
    # pip install ipython
    #
    # echo "Environment ready! Run 'source .venv/bin/activate' if not already activated."
  '';
}
