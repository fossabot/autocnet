===============================
AutoCNet
===============================

.. image:: https://badges.gitter.im/USGS-Astrogeology/autocnet.svg
   :alt: Join the chat at https://gitter.im/USGS-Astrogeology/autocnet
   :target: https://gitter.im/USGS-Astrogeology/autocnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/pypi/v/autocnet.svg
        :target: https://pypi.python.org/pypi/autocnet

.. image:: https://travis-ci.org/USGS-Astrogeology/autocnet.svg?branch=master
    :target: https://travis-ci.org/USGS-Astrogeology/autocnet

.. image:: https://coveralls.io/repos/USGS-Astrogeology/autocnet/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/USGS-Astrogeology/autocnet?branch=master

.. image:: https://img.shields.io/badge/Docs-latest-green.svg
    :target: hhttps://usgs-astrogeology.github.io/autocnet/
    :alt: Documentation Status

.. image:: https://badge.waffle.io/USGS-Astrogeology/autocnet.png?label=ready&title=Ready
 :target: https://waffle.io/USGS-Astrogeology/autocnet
 :alt: 'Stories in Ready'

Automated sparse control network generation to support photogrammetric control of planetary image data.

* Documentation: https://usgs-astrogeology.github.io/autocnet/

Installation Instructions
-------------------------
We suggest using Anaconda Python to install Autocnet within a virtual environment.  These steps will walk you through the process.

#. [Download](https://www.continuum.io/downloads) and install the Python 3.x Miniconda installer.  Respond ``Yes`` when
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fjlaura%2Fautocnet.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fjlaura%2Fautocnet?ref=badge_shield)

   prompted to add conda to your BASH profile.  
#. (Optional) We like to sequester applications in their own environments to avoid any dependency conflicts.  To do this:
   
   * ``conda create -n <your_environment_name> python=3 && source activate <your_environment_name>``
   
   Note, that you might want to specify either ``python=3.5`` or ``python=3.6``, depending on your requirements. Both are currently supported by autocnet.
#. Make the newly created environment the active one:

   * ``conda activate <your_environment_name>`` (or ``source activate`` on an older conda system)
   
#. Bring up a command line and add three channels to your conda environment-specific config file:
   
   * ``conda config --env --add channels conda-forge``
   * ``conda config --env --add channels menpo``
   * ``conda config --env --add channels usgs-astrogeology``
#. Finally, install autocnet: ``conda install autocnet``


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fjlaura%2Fautocnet.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fjlaura%2Fautocnet?ref=badge_large)