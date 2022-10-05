import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Построить нечёткую базу знаний для задачи определения цены видеокарты для
# компьютера ( потребность в графической обработке, электропотребление,
# разрешение экрана), проверить её на полноту и произвести нечёткий вывод для
# конкретных значений (выбрать случайным образом).


def main():
    graphical_need = ctrl.Antecedent(np.arange(0, 100), 'graphical_need')
    resolution = ctrl.Antecedent(np.arange(480 * 800, 4096 * 2160, 1000000), 'resolution')
    electricity = ctrl.Antecedent(np.arange(450, 750, 10), 'electricity')
    price = ctrl.Consequent(np.arange(500, 5750), 'price')

    price.automf(names=['low', 'medium', 'high'])

    graphical_need['low'] = fuzz.trapmf(graphical_need.universe, [0, 20, 30, 43])
    graphical_need['high'] = fuzz.trapmf(graphical_need.universe, [50, 60, 90, 100])

    resolution['low'] = fuzz.trapmf(resolution.universe, [480 * 800, 600 * 800, 580 * 900, 900 * 800])
    resolution['medium'] = fuzz.trapmf(resolution.universe, [1240 * 800, 1560 * 900, 1600 * 1000, 1920 * 1080])
    resolution['high'] = fuzz.trapmf(resolution.universe, [2000 * 2000, 2100 * 2100, 3840 * 1990, 4096 * 2160])

    electricity['low'] = fuzz.trapmf(electricity.universe, [450, 450, 500, 550])
    electricity['medium'] = fuzz.trapmf(electricity.universe, [560, 580, 590, 600])
    electricity['high'] = fuzz.trapmf(electricity.universe, [620, 650, 700, 750])

    graphical_need.view()
    resolution.view()
    electricity.view()

    rule1 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['low'], price['low'])
    rule2 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['high'], price['high'])
    rule3 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['medium'], price['low'])
    rule4 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['high'], price['low'])
    rule5 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['low'], price['high'])
    rule6 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['low'], price['high'])
    rule7 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['low'], price['medium'])
    rule8 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['medium'], price['medium'])
    rule9 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['high'], price['medium'])
    rule10 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['low'], price['high'])
    rule11 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['medium'], price['high'])
    rule12 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['high'], price['high'])
    rule13 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['medium'], price['low'])
    rule14 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['medium'], price['medium'])
    rule15 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['medium'], price['high'])
    rule16 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['high'], price['medium'])
    rule17 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['high'], price['low'])
    rule18 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['low'], price['high'])

    # rule1 = ctrl.Rule(graphical_need['low'] & electricity['low'], price['high'])
    # rule2 = ctrl.Rule(graphical_need['high'] & electricity['medium'], price['high'])
    # rule3 = ctrl.Rule(graphical_need['low'] & electricity['high'], price['medium'])
    # rule4 = ctrl.Rule(graphical_need['high'] & electricity['low'], price['high'])
    # rule5 = ctrl.Rule(graphical_need['low']& electricity['medium'], price['low'])
    # rule6 = ctrl.Rule(graphical_need['high'] & electricity['high'], price['medium'])

    # rule1 = ctrl.Rule(resolution['low'] & electricity['low'], price['medium'])
    # rule2 = ctrl.Rule(resolution['medium'] & electricity['low'], price['high'])
    # rule3 = ctrl.Rule(resolution['high'] & electricity['low'], price['high'])
    # rule4 = ctrl.Rule(resolution['low'] & electricity['medium'], price['low'])
    # rule5 = ctrl.Rule(resolution['medium'] & electricity['medium'], price['low'])
    # rule6 = ctrl.Rule(resolution['high'] & electricity['medium'], price['medium'])
    # rule4 = ctrl.Rule(resolution['low'] & electricity['high'], price['low'])
    # rule5 = ctrl.Rule(resolution['medium'] & electricity['high'], price['medium'])
    # rule6 = ctrl.Rule(resolution['high'] & electricity['high'], price['medium'])

    price_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15,
         rule16, rule17, rule18])
    # price_ctrl = ctrl.ControlSystem(
    #     [rule1, rule2, rule3, rule4, rule5, rule6])
    price_simulator = ctrl.ControlSystemSimulation(price_ctrl)

    price_simulator.input['graphical_need'] = 80
    price_simulator.input['resolution'] = 2000 * 1080
    price_simulator.input['electricity'] = 630

    price_simulator.compute()
    print(price_simulator.output['price'])

    graphical_need.view(sim=price_simulator)
    resolution.view(sim=price_simulator)
    electricity.view(sim=price_simulator)

    plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
