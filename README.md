# Présentation du sujet #

Pour le projet de modélisation,nous avons décidé de prendre le thème de l'intelligence artificielle par renforcement. Le principe d'une intelligence artificielle par renforcement réside dans son apprentissage en autonomie, l'IA apprend de ses expériences sur la base de récompense ou pénalité selon ses choix. Pour que notre IA puisse s'exercer, nous avons choisi le jeu du Tic Tac Toe qui nous paraissait la solution la plus à même d'être utilisé. Le but du jeu est très simple, deux joueurs s'affrontent sur une grille de 3X3, posant chacun leur tour le symbole qui leur correspond (X ou O). Pour gagner un des deux joueurs doit arriver à aligner 3 symboles identiques que ce soit de manière verticale, horizontale ou en diagonale.

En ce qui concerne nos perspectives futures pour cette IA, elles sont étendues. L'intelligence artificielle par renforcement offre un éventail de possibilités. On peut notamment penser aux voitures autonomes, optimisation logistique ou encore de la production, et cetera...

## Difficultés et solutions ##

Après avoir choisi notre sujet, il a fallu choisir à quel jeu notre IA allé jouer, mais surtout apprendre à jouer, nous avons donc proposé plusieurs jeux qui sont simples comme le jeu de Nim, puissance 4 , et cetera… Mais en évaluant leurs difficultés, on s'est tourné vers le Tic Tac Toe, qui était assez simple pour y implémenter une IA, et avec plein de stratégies différentes pour que celle-ci puisse apprendre.

Le second problème qui s'est posé est la non-connaissance du sujet. En effet, nous n'étions pas formés au développement d'une intelligence artificielle, ce qui nous a fallu une certaine période de documentation sur ce sujet pour atteindre ce résultat.

D'un autre côté, nous avons essayé de produire le graphe orienté pondéré des états possibles du tableau de jeu. Les sommets correspondent aux états possibles du tableau de jeu, les arêtes représentent la possibilité de passer d'un état à un autre et les poids correspondent aux valeurs stockées dans le fichier de stockage. Cependant, étant donné le nombre de sommets et d'arêtes, nous avons pu générer le graphe, mais sa représentation n'est pas lisible :



En effet, 3^9 = 19 683 états possibles du tableau puisque chaque case du tableau a 3 états possibles ('X', 'O', vide). Cependant, nous avons essayé de réduire ce nombre jusqu'à 6046 en enlevant les états impossibles (tableau remplit de symboles 'X', ...), mais nous n'avons pas eu le temps de l'implémenter.

Enfin, il nous a été assez difficile d'extraire les données les plus pertinentes pour réaliser des statistiques et des graphes qui le sont tout autant.

## Notre réalisation ##

Comme cité dans la présentation ci-dessus, nous avons réalisé notre projet sur la base du jeu Tic Tac Toe. Notre programme se sépare en différentes classes : Player, AI_RL, Human, Board et Game.

### Player ###

La classe Player permet de définir le joueur avec les variables de base comme le nom, le symbole (X ou O) et le type (IA ou Humain). De plus, la classe permet d'avoir les victoires, les pertes et les égalités ainsi que les valeurs d'apprentissages pour notre IA comme les mouvements effectués.

### AI_RL ###

La classe AI_RL, hérite de Player. En plus de tous les attributs de Player, cette classe inclut tous les paramètres liés à l'apprentissage de l'IA, tel que epsilon (probabilité d'exploration), le taux d'apprentissage, les mouvements joués ou encore les valeurs pour chaque état appris.

### Human ###

La classe Human hérite de Player et représente un joueur humain.

### Board ###

La classe Board permet la gestion de la grille de jeu. Elle détient notamment comme fonction, la réinitialisation de la grille et l'ajout des symboles. De plus, cette classe suit le déroulement de la partie en vérifiant les positions pour les items, si la position demandée est valide, si elle est déjà prise ou non. Enfin, elle permet de vérifier la fin de la partie, savoir s'il s'agit d'une égalité ou d'une victoire en vérifiant quel joueur a gagné.

### Game ###

La classe Game permet la gestion du jeu. Dans un premier temps, elle récupère une instance de Board pour déterminer la grille. Elle gère les deux joueurs de la parties ainsi que les premiers et seconds coups joués. En ce qui concerne la gestion des parties, la classe gère le tour par tour, vérifie la fin du jeu et le lance. Enfin, elle permet à l'utilisateur de choisir le mode de jeu et le nombre de parties qu'il souhaite.

## Où sont les mathématiques ? ##
Les mathématiques se retrouvent à plusieurs endroits de notre programme. Nous pouvons prendre l'exemple de la variable epsilon qui nous permet d'équilibrer les actions de l'intelligence artificielle par soit de l'exploration en jouant un coup aléatoire, soit l'utilisation des connaissances qu'elle a acquis tout le long de ses parties.

La plus grosse partie mathématiques de notre projet est la partie statistique. Elle comprend notamment un diagramme en secteurs représentant les taux de victoires, défaites et égalités ainsi qu'un second diagramme de secteurs qui lui représente les types de victoires (horizontale, verticale, diagonale). De plus nous générons deux graphiques matriciels permettant de représenter la grille de jeu en exposant le nombre de fois que chaque case a été choisi comme premier et second coups à l'aide de couleurs. Enfin nous générons deux graphiques linéaires exposant l'évolution des valeurs d'apprentissages au cours des différentes parties jouées pour le joueur 1 et le joueur 2.
