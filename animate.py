import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class AnimatedScatter(object):
    def __init__(self, user_locs, uav_locs):
        self.uav_locs = uav_locs
        self.user_locs = user_locs

        self.fig, self.ax = plt.subplots()

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, init_func=self.setup,
                                           frames=len(self.uav_locs), repeat=False)

    def setup(self):
        x_uavs = self.uav_locs[0][0]
        y_uavs = self.uav_locs[0][1]
        self.users = self.ax.scatter(self.user_locs[0], self.user_locs[1], s=20, c='gray')
        self.cov_circle = self.ax.scatter(x_uavs, y_uavs, s=6000, c='b', alpha=0.3)
        self.uavs = self.ax.scatter(x_uavs, y_uavs, s=20, c='r')
        self.uav_connection, = self.ax.plot(x_uavs, y_uavs, ':')

        self.ax.axis([0, 10, 0, 10])

        return self.cov_circle, self.users, self.uavs, self.uav_connection

    def update(self, i):
        x_uavs = self.uav_locs[i][0]
        y_uavs = self.uav_locs[i][1]
        self.cov_circle.set_offsets(np.c_[x_uavs, y_uavs])
        self.uavs.set_offsets(np.c_[x_uavs, y_uavs])
        self.uav_connection.set_data(x_uavs, y_uavs)

        return self.cov_circle, self.users, self.uavs, self.uav_connection


if __name__ == '__main__':
    user_locs = [[7.735914123868272, 2.4635156889219147], [2.9226870815078785, 7.567846555401347],
                 [3.3644034359513597, 7.1989610272638], [2.8733557885452945, 7.57725556054472],
                 [4.533489810278813, 6.672547353976236], [3.5549648061435435, 7.5635284892241454],
                 [2.8136470457974645, 8.02283018268535], [8.060910635495718, 3.5647419539559597],
                 [8.021106873357796, 3.291407606857911], [3.174159412971439, 7.792351193340396],
                 [3.8122369989956386, 8.258644901463006], [7.904082223819192, 2.556185517957582],
                 [3.0121865858100034, 7.214586889877178], [3.0305682952248723, 7.841580030240476],
                 [9.09278770326658, 2.301751832255931], [3.1627461914251165, 8.47778598480514],
                 [7.654169624137346, 2.8016232365720115], [3.352229369404728, 8.905258117772583],
                 [3.561302722309941, 8.032597108896901], [2.1092088713278043, 8.219483404324402],
                 [3.346647926784572, 7.877707788180249], [8.256464910209004, 2.8509535824486423],
                 [8.87240588210824, 2.6193965495524485], [3.3706257270468964, 8.09639084699817],
                 [3.054474789538894, 8.432866183398968], [7.846897993685814, 3.413987321303623],
                 [8.060079476240814, 3.3086015548537095], [8.15951954802855, 2.875314812261295],
                 [2.1389593050331266, 8.38537624718335], [7.9385548872406755, 2.532115282870466],
                 [8.150085159977914, 2.823875076753241], [2.038817875827271, 8.220611574429604],
                 [8.829901088554935, 3.3710220802886677], [3.245382185064875, 7.0814016121337975],
                 [8.599458939950754, 3.0925782087419718], [3.5923124533095234, 7.398338841417609],
                 [3.5183895157673923, 7.332409578734445], [2.982768130886955, 7.457386866581588],
                 [3.7271772213015897, 9.53088575288198], [7.664668856855485, 3.18878189316046],
                 [8.211747177032056, 3.0386700341742796], [8.731053968522486, 1.969929645251173],
                 [2.16051237736537, 8.604341722280815], [8.093280695494142, 3.205025823604128],
                 [3.28244414925356, 8.415504966265948], [7.449690411393539, 3.572361854919807],
                 [8.565814693725713, 3.7599084082110994], [7.664376934581591, 2.9936677005405494],
                 [8.450427974632206, 2.6581360704128336], [8.115047367682191, 3.3810055901560125],
                 [3.3089551104980073, 7.92989133670155], [3.6082203099041075, 8.525288147905506],
                 [3.4613348572701472, 6.86079316156862], [8.095457742333734, 4.0501275682394215],
                 [2.6380338794558407, 7.302081220644816], [7.828073162214462, 3.0217984284171235],
                 [1.6994126948872736, 8.865314821316975], [7.913785896224782, 2.561070791039314],
                 [7.622801029501674, 3.626434077616644], [7.88883592869482, 2.8996209655350014],
                 [3.6904346314748695, 8.14947373093481], [3.4216234962069403, 8.294197542552942],
                 [3.7849923957649647, 8.28623037655578], [3.8916291511111436, 9.376971224478314],
                 [4.769827994764478, 7.825955655599178], [8.14279366262713, 3.442570582135364],
                 [3.932019978811314, 7.798884495566787], [8.56688472116772, 2.4500543663429846],
                 [3.567666170962489, 8.73110946629508], [2.6870100046498013, 8.857155393363849],
                 [2.233561707981271, 8.118567703141068], [4.333261095838687, 8.777739688992353],
                 [7.895552883312611, 3.2933115955910988], [7.441344825682361, 3.117207848908546],
                 [8.080018534723916, 3.4380844605581125], [2.869401159664856, 7.928777889229599],
                 [2.5471583182215642, 8.631040824154098], [8.81217268183162, 2.694121793174962],
                 [7.689999578025935, 3.3490160170361096], [8.244259073268749, 2.962214143489472],
                 [4.139355527204234, 8.236608187602133], [7.277943097285205, 2.7477670685267745],
                 [3.1964655539858473, 7.948821107376905], [7.866055960186992, 3.265177733369093],
                 [7.838791397993246, 2.8079728226657923], [8.025403877388014, 2.6815021767153233],
                 [8.450795360296398, 3.251247169450934], [8.157817473620803, 1.9888993920879985],
                 [7.6564136499402, 2.57739717925064], [3.393932767633318, 8.168515964566412],
                 [7.626420853124581, 3.846227300513873], [7.42874090098893, 2.8253286387935614],
                 [8.099149860063385, 3.0595043229037295], [2.972913143513764, 6.8689593517076935],
                 [2.782918258537358, 6.295613565125003], [2.904488679272777, 7.916662068557633],
                 [8.419491706937253, 3.4655510406517784], [8.43270381466234, 1.8492306515598587],
                 [3.476048887914673, 7.77613888077093], [7.812357524954943, 2.680634796272889]]
    user_locs = np.array(list(zip(*user_locs)))



    # [[[x1, x2, x3, x4], [y1, y2, y3, y4]]]
    list_uav_locs = np.array([[[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]], [[1.0, 2.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0, 1.0]], [[0.0, 2.0, 1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 0.0, 1.0]], [[1.0, 2.0, 0.0, 0.0, 4.0], [2.0, 0.0, 0.0, 0.0, 1.0]], [[1.0, 3.0, 0.0, 1.0, 4.0], [3.0, 0.0, 1.0, 0.0, 2.0]], [[1.0, 4.0, 0.0, 1.0, 4.0], [4.0, 0.0, 2.0, 1.0, 1.0]], [[0.0, 3.0, 0.0, 0.0, 5.0], [4.0, 0.0, 3.0, 1.0, 1.0]], [[1.0, 3.0, 0.0, 1.0, 5.0], [4.0, 0.0, 4.0, 1.0, 0.0]], [[1.0, 3.0, 1.0, 1.0, 4.0], [3.0, 0.0, 4.0, 0.0, 0.0]], [[2.0, 3.0, 2.0, 1.0, 5.0], [3.0, 0.0, 4.0, 1.0, 0.0]], [[3.0, 2.0, 3.0, 1.0, 6.0], [3.0, 0.0, 4.0, 0.0, 0.0]], [[3.0, 1.0, 3.0, 1.0, 6.0], [4.0, 0.0, 3.0, 0.0, 1.0]], [[4.0, 2.0, 4.0, 1.0, 6.0], [4.0, 0.0, 3.0, 0.0, 2.0]], [[4.0, 2.0, 4.0, 1.0, 6.0], [3.0, 1.0, 2.0, 1.0, 1.0]], [[3.0, 1.0, 4.0, 0.0, 7.0], [3.0, 1.0, 1.0, 1.0, 1.0]], [[2.0, 0.0, 3.0, 0.0, 7.0], [3.0, 1.0, 1.0, 2.0, 2.0]], [[1.0, 0.0, 3.0, 0.0, 7.0], [3.0, 2.0, 0.0, 1.0, 1.0]], [[1.0, 0.0, 4.0, 0.0, 7.0], [2.0, 2.0, 0.0, 1.0, 2.0]], [[2.0, 0.0, 3.0, 0.0, 6.0], [2.0, 1.0, 0.0, 1.0, 2.0]], [[3.0, 1.0, 3.0, 0.0, 7.0], [2.0, 1.0, 0.0, 1.0, 2.0]], [[3.0, 0.0, 2.0, 1.0, 7.0], [1.0, 1.0, 0.0, 1.0, 1.0]], [[3.0, 0.0, 3.0, 2.0, 7.0], [0.0, 1.0, 0.0, 1.0, 0.0]], [[4.0, 0.0, 3.0, 3.0, 6.0], [0.0, 1.0, 1.0, 1.0, 0.0]], [[5.0, 0.0, 3.0, 4.0, 6.0], [0.0, 2.0, 0.0, 1.0, 1.0]], [[5.0, 0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 0.0, 2.0, 0.0]], [[5.0, 0.0, 2.0, 4.0, 6.0], [2.0, 4.0, 1.0, 3.0, 0.0]], [[6.0, 0.0, 1.0, 4.0, 7.0], [2.0, 3.0, 1.0, 2.0, 0.0]], [[6.0, 0.0, 0.0, 4.0, 7.0], [3.0, 3.0, 1.0, 1.0, 1.0]], [[5.0, 0.0, 1.0, 5.0, 6.0], [3.0, 4.0, 1.0, 1.0, 1.0]], [[5.0, 1.0, 1.0, 6.0, 5.0], [4.0, 4.0, 0.0, 1.0, 1.0]], [[5.0, 1.0, 1.0, 5.0, 4.0], [5.0, 5.0, 1.0, 1.0, 1.0]], [[5.0, 2.0, 1.0, 6.0, 4.0], [6.0, 5.0, 0.0, 1.0, 0.0]], [[5.0, 2.0, 0.0, 6.0, 4.0], [5.0, 6.0, 0.0, 2.0, 0.0]], [[6.0, 2.0, 0.0, 6.0, 4.0], [5.0, 7.0, 1.0, 3.0, 0.0]], [[7.0, 3.0, 0.0, 7.0, 5.0], [5.0, 7.0, 2.0, 3.0, 0.0]], [[8.0, 4.0, 0.0, 7.0, 5.0], [5.0, 7.0, 1.0, 4.0, 0.0]], [[9.0, 4.0, 0.0, 6.0, 5.0], [5.0, 8.0, 0.0, 4.0, 0.0]], [[9.0, 4.0, 0.0, 5.0, 5.0], [4.0, 9.0, 0.0, 4.0, 0.0]], [[10.0, 4.0, 0.0, 4.0, 4.0], [4.0, 10.0, 0.0, 4.0, 0.0]], [[10.0, 5.0, 0.0, 4.0, 4.0], [4.0, 10.0, 0.0, 5.0, 0.0]], [[10.0, 5.0, 0.0, 5.0, 5.0], [4.0, 9.0, 1.0, 5.0, 0.0]], [[9.0, 4.0, 0.0, 6.0, 5.0], [4.0, 9.0, 0.0, 5.0, 1.0]], [[9.0, 3.0, 1.0, 7.0, 4.0], [5.0, 9.0, 0.0, 5.0, 1.0]], [[10.0, 3.0, 1.0, 7.0, 3.0], [5.0, 10.0, 1.0, 4.0, 1.0]], [[10.0, 2.0, 1.0, 7.0, 2.0], [5.0, 10.0, 2.0, 5.0, 1.0]], [[10.0, 2.0, 0.0, 7.0, 2.0], [5.0, 10.0, 2.0, 6.0, 0.0]], [[9.0, 2.0, 0.0, 7.0, 2.0], [5.0, 10.0, 2.0, 5.0, 1.0]], [[9.0, 3.0, 1.0, 6.0, 2.0], [6.0, 10.0, 2.0, 5.0, 2.0]], [[10.0, 3.0, 2.0, 7.0, 3.0], [6.0, 9.0, 2.0, 5.0, 2.0]], [[10.0, 2.0, 1.0, 7.0, 3.0], [5.0, 9.0, 2.0, 6.0, 1.0]], [[9.0, 2.0, 2.0, 6.0, 3.0], [5.0, 10.0, 2.0, 6.0, 0.0]], [[10.0, 3.0, 2.0, 7.0, 2.0], [5.0, 10.0, 3.0, 6.0, 0.0]], [[10.0, 3.0, 2.0, 6.0, 2.0], [4.0, 9.0, 2.0, 6.0, 1.0]], [[10.0, 3.0, 1.0, 6.0, 2.0], [3.0, 8.0, 2.0, 5.0, 0.0]], [[9.0, 2.0, 1.0, 7.0, 1.0], [3.0, 8.0, 3.0, 5.0, 0.0]], [[9.0, 3.0, 2.0, 6.0, 1.0], [4.0, 8.0, 3.0, 5.0, 1.0]], [[10.0, 3.0, 3.0, 5.0, 2.0], [4.0, 9.0, 3.0, 5.0, 1.0]], [[10.0, 3.0, 3.0, 4.0, 2.0], [4.0, 8.0, 4.0, 5.0, 0.0]], [[10.0, 4.0, 4.0, 3.0, 2.0], [5.0, 8.0, 4.0, 5.0, 0.0]], [[10.0, 3.0, 4.0, 4.0, 3.0], [4.0, 8.0, 5.0, 5.0, 0.0]], [[10.0, 2.0, 5.0, 3.0, 4.0], [5.0, 8.0, 5.0, 5.0, 0.0]], [[9.0, 2.0, 5.0, 4.0, 3.0], [5.0, 7.0, 6.0, 5.0, 0.0]], [[9.0, 2.0, 5.0, 3.0, 4.0], [6.0, 8.0, 7.0, 5.0, 0.0]], [[9.0, 2.0, 6.0, 4.0, 4.0], [5.0, 7.0, 7.0, 5.0, 1.0]], [[8.0, 2.0, 6.0, 5.0, 4.0], [5.0, 6.0, 6.0, 5.0, 2.0]], [[7.0, 3.0, 7.0, 5.0, 3.0], [5.0, 6.0, 6.0, 4.0, 2.0]], [[7.0, 3.0, 7.0, 6.0, 4.0], [6.0, 5.0, 5.0, 4.0, 2.0]], [[6.0, 4.0, 8.0, 6.0, 4.0], [6.0, 5.0, 5.0, 3.0, 1.0]], [[6.0, 4.0, 9.0, 5.0, 3.0], [7.0, 6.0, 5.0, 3.0, 1.0]], [[7.0, 5.0, 9.0, 5.0, 3.0], [7.0, 6.0, 6.0, 4.0, 0.0]], [[6.0, 4.0, 8.0, 4.0, 3.0], [7.0, 6.0, 6.0, 4.0, 0.0]], [[6.0, 4.0, 7.0, 4.0, 3.0], [8.0, 5.0, 6.0, 3.0, 0.0]], [[6.0, 3.0, 6.0, 3.0, 4.0], [9.0, 5.0, 6.0, 3.0, 0.0]], [[5.0, 2.0, 6.0, 3.0, 4.0], [9.0, 5.0, 7.0, 4.0, 0.0]], [[5.0, 1.0, 7.0, 3.0, 5.0], [10.0, 5.0, 7.0, 5.0, 0.0]], [[6.0, 1.0, 8.0, 3.0, 5.0], [10.0, 4.0, 7.0, 4.0, 0.0]], [[6.0, 2.0, 7.0, 4.0, 4.0], [9.0, 4.0, 7.0, 4.0, 0.0]], [[5.0, 2.0, 7.0, 4.0, 3.0], [9.0, 3.0, 8.0, 5.0, 0.0]], [[5.0, 3.0, 7.0, 5.0, 4.0], [8.0, 3.0, 7.0, 5.0, 0.0]], [[5.0, 3.0, 6.0, 5.0, 3.0], [9.0, 4.0, 7.0, 4.0, 0.0]], [[4.0, 2.0, 7.0, 6.0, 3.0], [9.0, 4.0, 7.0, 4.0, 1.0]], [[3.0, 2.0, 6.0, 6.0, 2.0], [9.0, 3.0, 7.0, 5.0, 1.0]], [[4.0, 2.0, 6.0, 7.0, 3.0], [9.0, 2.0, 6.0, 5.0, 1.0]], [[4.0, 3.0, 6.0, 6.0, 3.0], [10.0, 2.0, 7.0, 5.0, 2.0]], [[5.0, 3.0, 7.0, 7.0, 2.0], [10.0, 3.0, 7.0, 5.0, 2.0]], [[5.0, 4.0, 8.0, 7.0, 1.0], [9.0, 3.0, 7.0, 6.0, 2.0]], [[4.0, 4.0, 7.0, 7.0, 2.0], [9.0, 2.0, 7.0, 5.0, 2.0]], [[4.0, 4.0, 7.0, 6.0, 2.0], [10.0, 3.0, 6.0, 5.0, 1.0]], [[5.0, 4.0, 7.0, 7.0, 1.0], [10.0, 4.0, 5.0, 5.0, 1.0]], [[5.0, 4.0, 7.0, 7.0, 2.0], [10.0, 5.0, 4.0, 6.0, 1.0]], [[5.0, 3.0, 7.0, 7.0, 3.0], [9.0, 5.0, 3.0, 5.0, 1.0]], [[5.0, 3.0, 8.0, 7.0, 4.0], [8.0, 6.0, 3.0, 6.0, 1.0]], [[6.0, 2.0, 7.0, 6.0, 3.0], [8.0, 6.0, 3.0, 6.0, 1.0]], [[6.0, 3.0, 6.0, 7.0, 3.0], [9.0, 6.0, 3.0, 6.0, 0.0]], [[7.0, 3.0, 6.0, 6.0, 4.0], [9.0, 7.0, 4.0, 6.0, 0.0]], [[6.0, 3.0, 5.0, 6.0, 4.0], [9.0, 6.0, 4.0, 7.0, 0.0]], [[6.0, 4.0, 5.0, 6.0, 3.0], [8.0, 6.0, 3.0, 6.0, 0.0]], [[7.0, 4.0, 5.0, 6.0, 3.0], [8.0, 7.0, 2.0, 5.0, 1.0]], [[7.0, 5.0, 6.0, 7.0, 2.0], [7.0, 7.0, 2.0, 5.0, 1.0]], [[8.0, 4.0, 6.0, 6.0, 3.0], [7.0, 7.0, 1.0, 5.0, 1.0]], [[9.0, 3.0, 5.0, 7.0, 2.0], [7.0, 7.0, 1.0, 5.0, 1.0]], [[8.0, 3.0, 6.0, 7.0, 2.0], [7.0, 6.0, 1.0, 6.0, 0.0]], [[9.0, 4.0, 7.0, 8.0, 1.0], [7.0, 6.0, 1.0, 6.0, 0.0]], [[9.0, 5.0, 7.0, 9.0, 1.0], [8.0, 6.0, 0.0, 6.0, 1.0]], [[9.0, 6.0, 6.0, 10.0, 2.0], [7.0, 6.0, 0.0, 6.0, 1.0]], [[8.0, 6.0, 7.0, 10.0, 2.0], [7.0, 5.0, 0.0, 6.0, 2.0]], [[8.0, 7.0, 7.0, 10.0, 2.0], [8.0, 5.0, 1.0, 7.0, 1.0]], [[8.0, 7.0, 6.0, 10.0, 1.0], [9.0, 4.0, 1.0, 8.0, 1.0]], [[8.0, 7.0, 6.0, 10.0, 0.0], [8.0, 3.0, 2.0, 9.0, 1.0]], [[9.0, 8.0, 6.0, 10.0, 0.0], [8.0, 3.0, 3.0, 10.0, 0.0]], [[10.0, 8.0, 6.0, 10.0, 0.0], [8.0, 4.0, 2.0, 9.0, 0.0]], [[10.0, 8.0, 6.0, 10.0, 0.0], [7.0, 5.0, 3.0, 10.0, 0.0]], [[10.0, 7.0, 5.0, 10.0, 0.0], [8.0, 5.0, 3.0, 10.0, 0.0]], [[9.0, 8.0, 6.0, 10.0, 0.0], [8.0, 5.0, 3.0, 10.0, 1.0]], [[9.0, 8.0, 7.0, 10.0, 0.0], [7.0, 4.0, 3.0, 10.0, 1.0]], [[8.0, 9.0, 7.0, 10.0, 1.0], [7.0, 4.0, 2.0, 10.0, 1.0]], [[9.0, 9.0, 7.0, 10.0, 2.0], [7.0, 5.0, 1.0, 9.0, 1.0]], [[8.0, 9.0, 8.0, 10.0, 3.0], [7.0, 6.0, 1.0, 9.0, 1.0]], [[7.0, 8.0, 9.0, 10.0, 4.0], [7.0, 6.0, 1.0, 8.0, 1.0]], [[8.0, 9.0, 10.0, 9.0, 5.0], [7.0, 6.0, 1.0, 8.0, 1.0]], [[7.0, 9.0, 10.0, 8.0, 4.0], [7.0, 5.0, 1.0, 8.0, 1.0]], [[7.0, 10.0, 10.0, 9.0, 5.0], [6.0, 5.0, 1.0, 8.0, 1.0]], [[7.0, 10.0, 10.0, 10.0, 5.0], [5.0, 4.0, 0.0, 8.0, 0.0]], [[7.0, 10.0, 9.0, 10.0, 5.0], [4.0, 5.0, 0.0, 8.0, 1.0]], [[7.0, 10.0, 10.0, 10.0, 5.0], [3.0, 4.0, 0.0, 8.0, 2.0]], [[7.0, 10.0, 10.0, 10.0, 6.0], [4.0, 3.0, 0.0, 8.0, 2.0]], [[8.0, 10.0, 10.0, 10.0, 6.0], [4.0, 3.0, 0.0, 8.0, 3.0]], [[9.0, 10.0, 9.0, 10.0, 6.0], [4.0, 2.0, 0.0, 9.0, 2.0]], [[10.0, 10.0, 9.0, 10.0, 5.0], [4.0, 3.0, 1.0, 8.0, 2.0]], [[10.0, 10.0, 9.0, 10.0, 5.0], [3.0, 2.0, 2.0, 8.0, 3.0]], [[10.0, 10.0, 8.0, 9.0, 4.0], [2.0, 1.0, 2.0, 8.0, 3.0]], [[10.0, 10.0, 9.0, 10.0, 5.0], [3.0, 1.0, 2.0, 8.0, 3.0]], [[10.0, 10.0, 9.0, 10.0, 5.0], [3.0, 2.0, 3.0, 8.0, 4.0]], [[9.0, 10.0, 8.0, 10.0, 4.0], [3.0, 1.0, 3.0, 7.0, 4.0]], [[9.0, 10.0, 7.0, 10.0, 4.0], [4.0, 0.0, 3.0, 6.0, 3.0]], [[10.0, 10.0, 7.0, 9.0, 3.0], [4.0, 1.0, 2.0, 6.0, 3.0]], [[10.0, 10.0, 6.0, 9.0, 3.0], [3.0, 0.0, 2.0, 5.0, 4.0]], [[10.0, 10.0, 6.0, 9.0, 3.0], [4.0, 1.0, 3.0, 4.0, 5.0]], [[9.0, 10.0, 5.0, 9.0, 2.0], [4.0, 0.0, 3.0, 3.0, 5.0]], [[9.0, 10.0, 6.0, 9.0, 2.0], [3.0, 1.0, 3.0, 2.0, 4.0]], [[9.0, 9.0, 6.0, 9.0, 1.0], [2.0, 1.0, 4.0, 3.0, 4.0]], [[9.0, 9.0, 6.0, 10.0, 1.0], [3.0, 2.0, 3.0, 3.0, 3.0]], [[8.0, 9.0, 7.0, 10.0, 1.0], [3.0, 1.0, 3.0, 4.0, 2.0]], [[7.0, 9.0, 6.0, 10.0, 1.0], [3.0, 0.0, 3.0, 4.0, 3.0]], [[6.0, 9.0, 5.0, 10.0, 0.0], [3.0, 0.0, 3.0, 5.0, 3.0]], [[6.0, 10.0, 4.0, 10.0, 0.0], [2.0, 0.0, 3.0, 5.0, 3.0]], [[7.0, 10.0, 3.0, 10.0, 1.0], [2.0, 0.0, 3.0, 4.0, 3.0]], [[6.0, 9.0, 3.0, 10.0, 1.0], [2.0, 0.0, 4.0, 5.0, 4.0]], [[6.0, 9.0, 2.0, 10.0, 1.0], [3.0, 1.0, 4.0, 4.0, 5.0]], [[5.0, 10.0, 1.0, 9.0, 2.0], [3.0, 1.0, 4.0, 4.0, 5.0]], [[5.0, 9.0, 1.0, 9.0, 2.0], [2.0, 1.0, 3.0, 3.0, 6.0]], [[4.0, 8.0, 1.0, 9.0, 2.0], [2.0, 1.0, 4.0, 4.0, 7.0]], [[4.0, 7.0, 1.0, 8.0, 1.0], [1.0, 1.0, 3.0, 4.0, 7.0]], [[4.0, 8.0, 1.0, 7.0, 1.0], [0.0, 1.0, 4.0, 4.0, 6.0]], [[4.0, 8.0, 1.0, 7.0, 0.0], [0.0, 0.0, 5.0, 5.0, 6.0]], [[5.0, 8.0, 1.0, 6.0, 0.0], [0.0, 1.0, 4.0, 5.0, 6.0]], [[6.0, 9.0, 1.0, 6.0, 0.0], [0.0, 1.0, 3.0, 4.0, 5.0]], [[6.0, 8.0, 1.0, 7.0, 1.0], [1.0, 1.0, 4.0, 4.0, 5.0]], [[6.0, 9.0, 1.0, 6.0, 1.0], [0.0, 1.0, 5.0, 4.0, 6.0]], [[6.0, 8.0, 1.0, 7.0, 0.0], [1.0, 1.0, 4.0, 4.0, 6.0]], [[6.0, 9.0, 0.0, 7.0, 0.0], [2.0, 1.0, 4.0, 5.0, 5.0]], [[5.0, 10.0, 0.0, 8.0, 0.0], [2.0, 1.0, 4.0, 5.0, 6.0]], [[6.0, 10.0, 1.0, 8.0, 0.0], [2.0, 2.0, 4.0, 4.0, 6.0]], [[7.0, 10.0, 1.0, 7.0, 0.0], [2.0, 3.0, 3.0, 4.0, 7.0]], [[7.0, 9.0, 1.0, 6.0, 0.0], [3.0, 3.0, 4.0, 4.0, 7.0]], [[6.0, 10.0, 1.0, 7.0, 0.0], [3.0, 3.0, 5.0, 4.0, 8.0]], [[7.0, 10.0, 1.0, 6.0, 0.0], [3.0, 3.0, 4.0, 4.0, 8.0]], [[7.0, 9.0, 0.0, 7.0, 0.0], [2.0, 3.0, 4.0, 4.0, 7.0]], [[7.0, 9.0, 0.0, 7.0, 0.0], [3.0, 4.0, 3.0, 5.0, 6.0]], [[7.0, 10.0, 0.0, 7.0, 0.0], [4.0, 4.0, 4.0, 6.0, 6.0]], [[7.0, 10.0, 0.0, 6.0, 0.0], [3.0, 5.0, 3.0, 6.0, 7.0]], [[8.0, 9.0, 0.0, 6.0, 1.0], [3.0, 5.0, 4.0, 5.0, 7.0]], [[9.0, 10.0, 0.0, 5.0, 0.0], [3.0, 5.0, 3.0, 5.0, 7.0]], [[9.0, 10.0, 0.0, 5.0, 0.0], [2.0, 4.0, 4.0, 4.0, 7.0]], [[8.0, 9.0, 1.0, 5.0, 0.0], [2.0, 4.0, 4.0, 3.0, 8.0]], [[7.0, 9.0, 0.0, 5.0, 1.0], [2.0, 5.0, 4.0, 2.0, 8.0]], [[6.0, 10.0, 0.0, 6.0, 0.0], [2.0, 5.0, 4.0, 2.0, 8.0]], [[6.0, 9.0, 0.0, 5.0, 0.0], [3.0, 5.0, 4.0, 2.0, 8.0]], [[6.0, 8.0, 1.0, 5.0, 0.0], [2.0, 5.0, 4.0, 3.0, 7.0]], [[5.0, 9.0, 1.0, 5.0, 0.0], [2.0, 5.0, 3.0, 2.0, 6.0]], [[5.0, 8.0, 1.0, 5.0, 0.0], [1.0, 5.0, 4.0, 1.0, 6.0]], [[6.0, 8.0, 0.0, 5.0, 0.0], [1.0, 6.0, 4.0, 0.0, 5.0]], [[5.0, 8.0, 0.0, 4.0, 0.0], [1.0, 7.0, 4.0, 0.0, 5.0]], [[5.0, 7.0, 1.0, 5.0, 1.0], [2.0, 7.0, 4.0, 0.0, 5.0]], [[5.0, 6.0, 0.0, 5.0, 2.0], [3.0, 7.0, 4.0, 1.0, 5.0]], [[5.0, 5.0, 0.0, 4.0, 2.0], [4.0, 7.0, 4.0, 1.0, 6.0]], [[5.0, 6.0, 0.0, 4.0, 3.0], [3.0, 7.0, 4.0, 2.0, 6.0]], [[5.0, 7.0, 0.0, 4.0, 3.0], [2.0, 7.0, 4.0, 1.0, 7.0]], [[5.0, 8.0, 1.0, 4.0, 3.0], [3.0, 7.0, 4.0, 2.0, 8.0]], [[5.0, 9.0, 1.0, 5.0, 3.0], [2.0, 7.0, 3.0, 2.0, 9.0]], [[6.0, 9.0, 1.0, 4.0, 2.0], [2.0, 6.0, 2.0, 2.0, 9.0]], [[6.0, 9.0, 2.0, 4.0, 3.0], [1.0, 7.0, 2.0, 1.0, 9.0]], [[6.0, 8.0, 1.0, 3.0, 2.0], [2.0, 7.0, 2.0, 1.0, 9.0]], [[5.0, 7.0, 1.0, 2.0, 1.0], [2.0, 7.0, 1.0, 1.0, 9.0]], [[4.0, 6.0, 1.0, 2.0, 0.0], [2.0, 7.0, 2.0, 2.0, 9.0]], [[3.0, 5.0, 0.0, 2.0, 0.0], [2.0, 7.0, 2.0, 3.0, 10.0]], [[4.0, 5.0, 0.0, 1.0, 0.0], [2.0, 6.0, 2.0, 3.0, 10.0]]])
    a = AnimatedScatter(user_locs, list_uav_locs)
    plt.show()