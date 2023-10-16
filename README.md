# Autonome Mikromobilität: Reinforcement Learning für ein Fahrrad in CARLA

Dieses Projekt geht darum, einem Fahrrad im CARLA-Verkehrssimulator mit Reinforcement Learning autonomes Fahren beizubringen. Dazu wird eine Tiefenkamera verwendet.
## Lernressourcen
<details>
  <summary>Mehr</summary>
  
### Einstieg Reinforcement Learning
https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

### Einstieg Stable Baselines 3
https://stable-baselines3.readthedocs.io/en/master/

</details>

## Installation 

<details>
  <summary>Details</summary>
  
### Anforderungen:
https://carla.readthedocs.io/en/latest/start_quickstart/#before-you-begin
- Windows oder Linux System
- mindestens 6GB GPU (für den CARLA-Server und Machine Learning)
- Python 2.7 oder Python 3.0 für Linux, Python 3.0 für Windows
- mindestens pip 20.3 (bzw. pip3 20.3)

Außerdem sollte numpy installiert sein.

Windows: 

    pip3 install --user pygame numpy
    
Linux: 
    
    pip install --user pygame numpy &&
    pip3 install --user pygame numpy
    
### CARLA:
https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation

Einrichtung des APT-Repositories:

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
    sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

CARLA installieren: 

    sudo apt-get update # Update the Debian package index
    sudo apt-get install carla-simulator # Install the latest CARLA version, or update the current installation
    cd /opt/carla-simulator # Open the folder where CARLA is installed

CARLA Client-Library installieren (in Virtual Environment empfohlen): 
Download von  https://pypi.org/project/carla/ (kompatibel mit Python 2.7, 3.6, 3.7, und 3.8.)

    pip3 install carla  

### Stable Baselines 3 
https://stable-baselines3.readthedocs.io/en/master/guide/install.html#stable-release
Installation inklusive Tensorboard und OpenCV:

    pip3 install stable-baselines3[extra] 
    
### Gymnasium

    pip3 install gymnasium
    
### Verwendete Versionen:
- numpy 1.23.5
- pygame 2.1.3
- carla 0.9.13
- gymnasium 0.28.1
- stable-baselines3 1.8.0

</details>


## Training 

<details>
  <summary>Details</summary>
Es sollten sich alle Python Dateien in einem Ordner (hier: autonomous_bike) befinden.

### CARLA starten

    cd /opt/carla-simulator
    ./CarlaUE4.sh
    
### Environment checken
Bei Anpassungen der Environments sollten diese vor dem Training auf Implementierungsfehler gecheckt werden.

    cd ./autonomous_bike
    python3 check_env.py
    python3 doublecheck_env.py
    
### Training starten 
Je nach importierter Environment wird ein anderer Agent trainiert (go_to_goal_env.py, go_to_goal_env_2.py, collision_avoidance_env.py oder  collision_avoidance_env_modified.py).

    python3 train.py
   
 Um alte Models weiterzutrainieren oder eine eigene Learning Rate zu verwenden, muss train.py angepasst werden.

### Nutzung von Tensorboard
Das Training kann im Anschluss über Tensorboard analysiert werden. Dafür muss im Ordner in dem der Ordner logs angelegt wurde, folgender Befehl im Terminal ausgeführt werden.

    tensorboard --logdir=logs

Über den zurückgegebenen Link lässt Tensorboard sich öffnen.

### Models laden
Um trainierte Models zu laden muss load_model.py entsprechend angepasst und folgender Befehl ausgeführt werden.

    python3 load_model.py
    
</details>

