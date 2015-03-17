# egcm_data.R
# Copyright (C) 2014 by Matthew Clegg
# Data sets used for calibrating the Engle Granger cointegration tests
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# A copy of the GNU General Public License is available at
# http://www.r-project.org/Licenses/
# The following table was generated using the call
# egc_adf_qtab <- egc_quantile_table("adf")


import numpy as np
import pandas as pd


# The following table was generated using the call
# egc_adf_qtab <- egc_quantile_table("adf")
egc_adf_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25, -6.82250219242378, -4.46661052588325, -4.00475599937948,
               -3.66220748383464,
               -3.28216656346919, -2.86590314679678, -2.16913368045838,
               -1.50075662675211, -1.09869456855476, -0.707376464101655, -0.332468251332744,
               0.107652360327496, 1.00709564081704, 1.91846330041794],
        '50': [50, -5.90031369372017, -4.24974811950817, -3.89196306810039,
               -3.59026093887678, -3.25586397493684, -2.89293708645451,
               -2.25351483865642, -1.63274187325847, -1.24282567813493, -0.886990250678287,
               -0.557740434091227, -0.142320559436172, 0.758470475525642, 1.32529566972668],
        '100': [100, -5.35423335387404, -4.16243361491686, -3.83602412929703,
                -3.55493732070356, -3.25821675368631, -2.90647707985012,
                -2.28872436582548, -1.68389591712136, -1.3247114283679,
                -0.98259702440316,
                -0.661702171343778, -0.272200443099633, 0.571558557411948,
                1.26134849462017],
        '250': [250, -5.32169658333163, -4.11068853767939, -3.81940000984988,
                -3.56119634309084, -3.26424779449212, -2.92597765126574,
                -2.31492857507422,
                -1.72444096801193, -1.38783722528013, -1.07095161034388,
                -0.779374411552248,
                -0.423320784086461, 0.40567187496224, 1.024742338797],
        '500': [500,
                -5.24438771322382, -4.1046678117876, -3.80394774580977,
                -3.55339441563194,
                -3.27695767692977, -2.94440908560947, -2.33938602148616,
                -1.74581654896007,
                -1.4132857939648, -1.1169379497244, -0.830141162033237,
                -0.465778748069809,
                0.385094864798775, 0.964119035978792],
        '750': [750, -5.25983185847632,
                -4.12933224960047, -3.83970653883534, -3.57291092601257,
                -3.28898463982481,
                -2.94582095357127, -2.34103700922763, -1.74956185549751,
                -1.41911350392386,
                -1.11329138497356, -0.837265137997805, -0.501527433716493,
                0.207037398489134,
                0.853874254435031],
        '1000': [1000, -5.31863097152306, -4.14687976118349,
                 -3.82721765117813, -3.57658709189777, -3.29530031385029,
                 -2.95266445456483,
                 -2.34541912903951, -1.74677020913492, -1.4303717434034,
                 -1.13415927195089,
                 -0.855675386839316, -0.517178696611797, 0.194504972589135,
                 0.953935200801842],
        '1250': [1250, -5.34381676607384, -4.10022572980013, -3.82008827402893,
                 -3.57080284633046, -3.29051306783958, -2.95763482283479,
                 -2.34815790334682,
                 -1.75337613742464, -1.43607541465393, -1.13530579667183,
                 -0.867651555662172,
                 -0.52451703983386, 0.163175391030527, 0.79066390181105],
        '2500': [2500,
                 -5.48263498988541, -4.10626409894068, -3.81637022589474,
                 -3.57405918222471,
                 -3.28863948464811, -2.96305857836569, -2.35770540383653,
                 -1.76998261937848,
                 -1.45053398476124, -1.1535106961009, -0.87911223343783,
                 -0.526093269639521,
                 0.271375890964498, 0.822501041543799],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_pp_qtab <- egc_quantile_table("pp")
egc_pp_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9,
                  0.95, 0.975, 0.99, 0.999, 0.9999],
     '25': [25,
            -31.7483317260169, -22.4259942143006, -20.0891471142396,
            -18.2228875467491,
            -16.2900282564813, -14.0290911430216, -9.98925085234759,
            -6.41035091322298,
            -4.70746367352161, -3.3545521937257, -2.20449603515498,
            -1.00333103816288,
            1.76278039927971, 4.36583517280819],
     '50': [50, -39.8204238377295,
            -27.3302057249438, -24.0799945780677, -21.6516578455852,
            -19.0697266343428,
            -16.0045124615486, -10.8304998604655, -6.71114279673032,
            -4.95283812295386,
            -3.58773892769175, -2.53250950500307, -1.33650377090982,
            0.849883907175838,
            2.87260613428669],
     '100': [100, -44.7864231243006, -30.638876291843,
             -27.024277485246, -23.9091867343539, -20.5387166669581,
             -16.8078987859275,
             -11.0424387103879, -6.67592689143081, -4.87340746087305,
             -3.59267645284015,
             -2.54631521611987, -1.48163658965051, 0.654286425103889,
             2.45308734981177],
     '250': [250, -52.3813922346657, -32.7257688261613, -28.2514965979308,
             -24.7843612597892, -21.193356294857, -17.2952885039841,
             -11.1154112538718,
             -6.73989823169638, -4.92626694694195, -3.64195105982085,
             -2.63349537765853,
             -1.5885165993875, 0.382882246362119, 1.72727581682196],
     '500': [500,
             -52.4313893755687, -32.6520179408335, -28.2585583397316,
             -24.6814868556535,
             -21.0347391176349, -17.0942430028448, -10.9731074618946,
             -6.60661454001619,
             -4.87079674365988, -3.66571472079707, -2.63806781366398,
             -1.60795720185688,
             0.474461376434679, 1.7231901818137],
     '750': [750, -51.7027126407095,
             -33.1620272058072, -28.3591110876713, -24.7988360099677,
             -21.0460064345614,
             -16.9322970862411, -10.8738480586768, -6.58818302982247,
             -4.83041867050955,
             -3.59149192513659, -2.61647545777101, -1.51292326090877,
             0.644757851844955,
             2.48866850121809],
     '1000': [1000, -54.4030728852552, -32.712384272494,
              -28.3999735902766, -24.968962543997, -21.1185117281465,
              -17.0447483918261,
              -10.8895595879606, -6.55758523630802, -4.80567543136177,
              -3.56763756304767,
              -2.57884471059726, -1.42233865489255, 0.474594978280921,
              2.06862897693011],
     '1250': [1250, -58.0986707297663, -33.0461480510437, -28.5125014119486,
              -24.8559511482087, -21.1071169863818, -17.0973790437607,
              -10.9683521659909,
              -6.57573587613659, -4.8495786821022, -3.63118603392207,
              -2.66296753810603,
              -1.59045686730992, 0.408910258628224, 2.46146883303944],
     '2500': [2500, -53.6421196025365, -32.5390612193389, -27.8551479695678,
              -24.3774680094461, -20.7899114890461, -16.9111234650402, -10.9055637754689,
              -6.60540322577147,
              -4.8653314977783, -3.60531304485548, -2.63282041695351,
              -1.50078156941867,
              0.592412700964834, 2.89693019375655],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_pgff_qtab <- egc_quantile_table("pgff")
egc_pgff_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-0.336625831999346, 0.0341437189994142, 0.143854756626506, 0.227942822537199,
0.321875762327391, 0.430157943286197, 0.616450377657225, 0.762759885088315,
0.82361806179326, 0.866940512284584, 0.899155615642256, 0.934900153650464,
1.00246211010138, 1.05047797561818],
        '50': [50, 0.220627836953644,
0.448634437995673, 0.514235633116768, 0.571095688677306, 0.632523754087507,
0.697606185429773, 0.803745405221187, 0.883213445856658, 0.915783884604886,
0.938039550199234, 0.957013147125422, 0.975923560598361, 1.01644861773643,
1.04196578449547],
        '100': [100, 0.541652148126685, 0.712994671815014,
0.747472775368399, 0.777222175184613, 0.810287530989747, 0.845281726392217,
0.900971180378913, 0.942387342535367, 0.959658313907702, 0.971236226780725,
0.980263097643523, 0.990058680715472, 1.00772162129372, 1.02230585650179],
        '250': [250, 0.796887373191436, 0.87644234791977, 0.89398291812434,
0.907551318142438, 0.921946440048468, 0.936960981640209, 0.960197179275122,
0.977145020492887, 0.984113175240602, 0.989014265711227, 0.992648985719689,
0.996589560970954, 1.00422289236937, 1.00981839336385],
        '500': [500,
0.906471394939692, 0.938443130778873, 0.947340862521741, 0.953973549191814,
0.961015371233118, 0.968651861560137, 0.980224315770123, 0.988669425830378,
0.991983402705577, 0.994422930741069, 0.996277428012133, 0.998371861509028,
1.00210286323482, 1.00474384238835],
        '750': [750, 0.929012686308371,
0.958019758613622, 0.96424798196568, 0.968767071025123, 0.973765639015405,
0.978963328796586, 0.986769637760877, 0.992495831119869, 0.994822100137963,
0.996421963380021, 0.997694650337578, 0.999024753535889, 1.00178096292395,
1.00333529846727],
        '1000': [1000, 0.949176685091427, 0.969203355354484,
0.973381143537786, 0.976701237235953, 0.980345880136921, 0.984190946880614,
0.990107914093208, 0.994344287184784, 0.996067492560003, 0.997292230446198,
0.998227124217666, 0.999197584772469, 1.00094716404021, 1.00270750718348],
        '1250': [1250, 0.960643156183032, 0.975146939684992, 0.978682825560059,
0.981431908867788, 0.984285583607903, 0.987292545812951, 0.992055628457898,
0.995450138647364, 0.996810989623488, 0.99777743131819, 0.998559289716415,
0.999300771753356, 1.00098023261994, 1.00232131446558],
        '2500': [2500,
0.979239143080471, 0.987594556555884, 0.989307688819376, 0.990671712084724,
0.992159172127179, 0.993692063156018, 0.996040818179021, 0.997734558408831,
0.998422620884948, 0.998914003067803, 0.999291096260776, 0.999692009809386,
1.00042127303711, 1.00133189611659],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_joe_qtab <- egc_quantile_table("jo-e")
egc_joe_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-37.4934491481087, -23.4752500251861, -20.7322351509806, -18.4802462385388,
-16.0904351483193, -13.5967528699361, -9.72782001868844, -6.85330218916773,
-5.67715127234161, -4.86414622392108, -4.2424163183183, -3.59728147515604,
-2.6331400775029, -1.93339435261407],
        '50': [50, -34.4383406226246,
-21.5843795460596, -18.9497918614636, -16.9014607734677, -14.7380598332874,
-12.4963014673808, -8.93765721819998, -6.30730843381653, -5.23804065964026,
-4.45726352911189, -3.89418248928871, -3.30788625722547, -2.3253363699569,
-1.83973277304182],
        '100': [100, -31.4969937222486, -20.6542048961051,
-18.2994013572307, -16.3016078616509, -14.3018906386804, -12.1089714262841,
-8.69332514602511, -6.11081103857875, -5.00977087514907, -4.27494697678329,
-3.69821816214648, -3.14606616699598, -2.29523504952876, -1.88177947492758],
        '250': [250, -29.9886203544917, -20.3134182049498, -17.7958631304333,
-15.9003842383322, -13.9392094994162, -11.8506643765848, -8.49972915146836,
-5.99143403235332, -4.94647714760393, -4.2292217784859, -3.67847415991356,
-3.12273027260759, -2.18613631874027, -1.65620389938301],
        '500': [500,
-31.0799734994848, -20.3476972506917, -17.8641226163083, -15.984269524298,
-13.9885808720719, -11.8609093357509, -8.47264259127889, -5.96103482546849,
-4.91823850171487, -4.18169641700738, -3.63141265114147, -3.06808234648939,
-2.21626106403673, -1.64730084697278],
        '750': [750, -29.7711360611772,
-20.2485974761581, -17.9400968340255, -16.0018615584896, -14.0424937323286,
-11.8515113665148, -8.44217565186761, -5.92367629729155, -4.89155291683393,
-4.14127061751557, -3.57756688544588, -3.0410906729227, -2.10959025824598,
-1.72552476877069],
        '1000': [1000, -29.8691782696184, -20.23651212437,
-17.9051159103113, -16.0115740479624, -13.9679312360707, -11.7487752998559,
-8.42039292462495, -5.88719641677412, -4.85645488013636, -4.13790343739176,
-3.59785138916529, -3.0380644069827, -2.17965907294437, -1.64956499964293],
        '1250': [1250, -32.4070442032071, -20.3053859149176, -17.8479733284502,
-16.0097134307693, -14.0199581926918, -11.8345750857438, -8.46600371563483,
-5.90968973735435, -4.88283780987849, -4.15764219129007, -3.59892608622552,
-3.07077442036504, -2.21070605459954, -1.54299694506694],
        '2500': [2500,
-30.3095709948853, -20.245554884956, -17.8867818774405, -16.0047058240053,
-13.9945246714598, -11.7875259069063, -8.43968864047374, -5.92373337410219,
-4.87901776894278, -4.16056440196444, -3.62883846279096, -3.07973101793951,
-2.10998984940825, -1.43505686885504],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_jot_qtab <- egc_quantile_table("jo-t")
egc_jot_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-43.6039139855407, -28.6829709794164, -25.6300537603927, -23.3079359185719,
-20.643739189464, -17.7884740594005, -13.1348239053396, -9.58693660153554,
-8.05909112818845, -7.01406411497141, -6.19569951461884, -5.35591475011705,
-4.08260696696836, -3.31049871144232],
        '50': [50, -38.4752049971958,
-26.8080988779882, -23.9437700481207, -21.5574845593767, -19.0774684501909,
-16.5045018454215, -12.183631307656, -8.84373681572512, -7.46100621867351,
-6.47530477837974, -5.69218935468569, -4.89007873490613, -3.66579155674075,
-2.7324831220996],
        '100': [100, -40.2124175203202, -25.5893372632967,
-23.0323211816574, -20.810618288626, -18.5040229104433, -15.9183860115843,
-11.74611718597, -8.55756459180621, -7.1880271146337, -6.1612913901399,
-5.4259638635224, -4.66717848573724, -3.64213982185279, -2.82147037506989],
        '250': [250, -36.8363870121499, -25.503723090232, -22.7913067921628,
-20.5647324983675, -18.3227705275422, -15.7148313468566, -11.5691663072761,
-8.37606828061385, -7.01749168898447, -6.07378903078064, -5.33387231670868,
-4.55762374951882, -3.37367431197166, -2.71177217407318],
        '500': [500,
-35.8931079990305, -25.5067935105137, -22.6971799724303, -20.4943323346414,
-18.0772809282935, -15.5533042944476, -11.4923511862326, -8.32760076448148,
-7.00568646305856, -6.07714694733917, -5.32811847017675, -4.58102351459653,
-3.25159378652592, -2.65054723217171],
        '750': [750, -37.745032170643,
-25.0822599618117, -22.4732901668475, -20.3703679268076, -18.0870793164142,
-15.5682318967508, -11.4780249865322, -8.33286518403125, -7.00996697530282,
-6.04396418081035, -5.30102486235099, -4.57043493336911, -3.35455895999861,
-2.73356562079151],
        '1000': [1000, -36.2344573655043, -25.1038801150724,
-22.3998867232976, -20.206581784596, -17.9432295463599, -15.4644455837395,
-11.4531807056099, -8.33200153188878, -6.98751115066501, -5.99846251864034,
-5.30359410405053, -4.55912626375865, -3.33973427265424, -2.69388481757016],
        '1250': [1250, -36.422058957083, -25.1248808353849, -22.4460546913756,
-20.3446739203963, -18.0567540031894, -15.5546815514663, -11.4659570172447,
-8.29301400579434, -6.94723871532745, -5.97833367858986, -5.20389564066867,
-4.44983350810362, -3.32349110694732, -2.64683960242408],
        '2500': [2500,
-35.6379697758194, -25.2733069974919, -22.5247104652215, -20.2349350303678,
-18.018940187706, -15.4994530355328, -11.4738335040917, -8.29267196408465,
-6.97256431918771, -6.01373746763148, -5.29545588131238, -4.58138773087048,
-3.30109427170537, -2.42493858524213],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_ersp_qtab <- egc_quantile_table("ers-p")
egc_ersp_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
1.62524409776054e-07, 0.000993351217457755, 0.00632761422919328,
0.0249541713400963, 0.100253417145221, 0.418009322648233, 2.74077437917608,
11.8806257428513, 26.4527501186186, 50.121416245285, 87.4788254786966,
154.044526454243, 482.515677121453, 1057.59828306248],
        '50': [50,
4.41794940706963e-06, 0.0336334027403853, 0.158165985952471,
0.452392884816047, 0.97261682326569, 1.85027936383228, 4.72929871946086,
12.6897052690977, 22.5058316159347, 35.929045705844, 53.7546692149607,
85.7649714447181, 206.542652092838, 345.09085580573],
        '100': [100,
0.0619721273346449, 0.773203593988022, 1.1139559460748, 1.48875625449004,
2.0335260175412, 2.96866136348471, 6.4919829149133, 15.8510632535666,
26.1910387762326, 38.9669225289442, 56.0924147744943, 83.9700427535428,
162.266700361115, 281.921761751524],
        '250': [250, 0.55528634965954,
1.23137663481992, 1.57480118078469, 1.94753752564035, 2.51853403907714,
3.61084771611864, 7.89655166213118, 19.383586807307, 31.2095674719374,
45.6997855930344, 62.1238728131958, 88.3597119501477, 166.937708487234,
263.921546782893],
        '500': [500, 0.639334997026308, 1.30343258621287,
1.62016826096752, 2.01087268057896, 2.61945158602907, 3.75029646128193,
8.23426950014149, 20.4772688421125, 32.9302712128018, 47.7929142735086,
65.0872316662308, 89.3623155688477, 167.423512438777, 239.86696480845],
        '750': [750, 0.675840415460787, 1.30657907092168, 1.64810139626585,
2.04212723356586, 2.66056776532775, 3.78753790421503, 8.53922847595784,
21.1541280645958, 33.7371457369047, 48.9875802648531, 66.9431017376182,
94.0032520809409, 165.723932695121, 245.598387008322],
        '1000': [1000,
0.672525283346607, 1.31610886967128, 1.65049053857724, 2.03580193850619,
2.64001441933625, 3.82888572913011, 8.62801341605654, 21.5655364609746,
34.2497810388054, 49.4081675180479, 67.1088208929885, 95.804137908048,
167.932906806478, 291.483436589401],
        '1250': [1250, 0.7407466176107,
1.3337353248466, 1.67724864033851, 2.07287011315128, 2.70810479663192,
3.85021716960402, 8.6759735666651, 21.4399104286228, 34.367864784182,
49.8731322659698, 67.9241591648535, 95.2495997369313, 176.691645517416,
244.034782349468],
        '2500': [2500, 0.675518625055808, 1.33368882838333,
1.68265645154853, 2.06039289391705, 2.68819749443941, 3.86333765228147,
8.71019616325151, 21.5463829066756, 35.0497499892926, 50.3445699503282,
69.0720626677329, 94.1117432628981, 167.731713289658, 231.630106373227],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_ersd_qtab <- egc_quantile_table("ers-d")
egc_ersd_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-5.10048956847502, -3.30687485693397, -2.90155974190788, -2.59129558300045,
-2.27276240072175, -1.92213650552829, -1.34798299473902, -0.838329861629613,
-0.509185593064544, -0.195300939641466, 0.108914109087463, 0.482149591268087,
1.32699563522915, 2.10728039324104],
        '50': [50, -4.6199438788185,
-3.21676499222889, -2.8686006810382, -2.59803884713455, -2.29576353217148,
-1.94583374315336, -1.33829556692924, -0.781175098555971, -0.397080658949842,
-0.0296455279811186, 0.32631220386273, 0.734690136808975, 1.62622811094037,
2.24605507511686],
        '100': [100, -4.51297528529818, -3.17156669441703,
-2.83736242414913, -2.55094926095896, -2.24977219003744, -1.88595137575393,
-1.24474595982246, -0.638191937730234, -0.233009600573567, 0.160302569814843,
0.51339966804765, 0.951450453947234, 1.81236847378032, 2.69868923028809],
        '250': [250, -4.37525803985682, -3.11239840487866, -2.80614575845577,
-2.53816683428338, -2.20592284886742, -1.83484806078488, -1.1468850571759,
-0.491236431122179, -0.0764269107363089, 0.312863660226734, 0.666606184291284,
1.10400057132307, 1.9447657288027, 2.77038477905867],
        '500': [500,
-4.25804077548967, -3.12780400002265, -2.81162151984338, -2.52181493428759,
-2.18877158176733, -1.7876628512075, -1.08682320198154, -0.405262455957855,
0.0217186030751472, 0.389864188676898, 0.738630355309322, 1.15496529052512,
2.05416971254848, 2.90495065530857],
        '750': [750, -4.36147063522307,
-3.14245449845762, -2.78950724295031, -2.48675051561575, -2.1653912230568,
-1.78073016343705, -1.06913732108122, -0.381437995991238, 0.0484265194036463,
0.434833031029738, 0.77225944105774, 1.18255474811806, 1.99386245001663,
2.80529292874148],
        '1000': [1000, -4.50739170629214, -3.13712650619975,
-2.80431314786342, -2.50954785628484, -2.18074731078809, -1.78467925508109,
-1.07259971623705, -0.382209469861783, 0.0576012359080366, 0.435548812016741,
0.800719533205871, 1.18775043814421, 2.08391466082401, 2.78987232526501],
        '1250': [1250, -4.52635229954304, -3.12054809254334, -2.8022823402926,
-2.5028174903535, -2.16809873100608, -1.77226225298197, -1.06144666113106,
-0.368677919111695, 0.0550234663325856, 0.445648528283737, 0.807052347089671,
1.19924309600821, 1.96541372814745, 2.62092875436772],
        '2500': [2500,
-4.47168620669166, -3.13545292848152, -2.78588889223927, -2.4880731613059,
-2.15658084744067, -1.77365567712487, -1.04468871221604, -0.353635563847591,
0.0725063965562995, 0.464447482050564, 0.817423489553938, 1.21452650295202,
2.16613650569432, 2.8166674020312],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_spr_qtab <- egc_quantile_table("sp-r")
egc_spr_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-52.6580674935556, -22.3861520557875, -17.3864412906388, -14.421268812453,
-11.7167753282881, -9.29292285283269, -4.79260434274902, 0.887548308645351,
4.01376339096833, 7.02231750883232, 10.4327290224018, 15.2686651172952,
27.2684638444846, 38.4394696796664],
        '50': [50, -61.4048593121644,
-26.061017492712, -22.0403196103959, -19.2546202797511, -16.7412908851307,
-14.226477705248, -9.22334190406295, -4.13202874459074, -1.40276812028152,
0.592906112391562, 2.20918217496902, 4.3072581136573, 11.1808081177222,
20.2478559929431],
        '100': [100, -50.3039002015622, -29.7328726027708,
-26.729419424891, -24.2759372390848, -21.397555121129, -17.8587238673482,
-11.0925227673756, -5.4046402185745, -3.14839559931649, -1.37810591698949,
-0.0718170814128574, 1.45388853258661, 4.16593145701825, 6.41447232441966],
        '250': [250, -55.4541916697354, -37.2633198539768, -32.321570871558,
-28.1076003621881, -23.4690564850126, -18.4305177297307, -10.5321701441243,
-5.21312276162628, -3.23732701223692, -1.89619562108202, -0.877880133453622,
0.290011112144745, 2.73355064701423, 4.28148964963558],
        '500': [500,
-65.196373572361, -38.5196597911918, -32.6934163091834, -27.6488708555601,
-22.6425190442907, -17.3691458046934, -9.73585377523323, -4.95817126843802,
-3.29752837863405, -2.16705673928955, -1.3550483851403, -0.427178605258981,
1.54421759509182, 3.04795877851944],
        '750': [750, -66.2992118292477,
-37.4258899749959, -31.4471300098562, -26.6263907489262, -21.7164895305755,
-16.7450987696474, -9.44524927775992, -4.96136619440135, -3.38343701051509,
-2.39004400329423, -1.6405952088073, -0.899695677497808, 1.01592238207383,
2.62243764809039],
        '1000': [1000, -63.3941453684708, -36.8083513432148,
-30.7371264951546, -26.2579808522556, -21.3233625045404, -16.3143971164037,
-9.26471707284782, -4.94645341482234, -3.4552037954847, -2.49010210001177,
-1.78806974309296, -1.0304334433519, 0.43986080730562, 2.46811417886369],
        '1250': [1250, -66.3973686585401, -36.3669268158086, -29.9693035813389,
-25.525069041783, -20.7584281320636, -16.0088474590749, -9.09579226688802,
-4.91825731265008, -3.46908764338723, -2.57307710737357, -1.94111393050574,
-1.27888612803405, 0.0158725164487333, 1.71854940369779],
        '2500': [2500,
-57.280232462311, -34.3510602355879, -28.2346144973038, -24.019011625483,
-19.7639564036047, -15.3006473206934, -8.95370586432994, -5.00062203171719,
-3.65859256643428, -2.80814433773019, -2.21352451614216, -1.67558352731201,
-0.826952287157412, -0.0526180736698275],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_bvr_qtab <- egc_quantile_table("bvr")
egc_bvr_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
0.00175316032383965, 0.00372824609873825, 0.00474477225593607,
0.00605469351971975, 0.00812844997687993, 0.0115842364885377,
0.0236518010608536, 0.0515055182026042, 0.0679116332488497, 0.078356862924516,
0.0849650741569281, 0.090225525270155, 0.0962490330322663, 0.0985873233848373],
        '50': [50, 0.00125878578262598, 0.00348407042704959, 0.00453078060718951,
0.00582061896831429, 0.00776085926148126, 0.0112502774429388,
0.0230665927673575, 0.0513883374845921, 0.0676540246205908, 0.0783421750552322,
0.0852850990434269, 0.0905302715688617, 0.0961929557175782, 0.0985464434960829],
        '100': [100, 0.00126646991292459, 0.00341968283729603, 0.00446998420728161,
0.00573010282121679, 0.00767810999346112, 0.0111811961826295,
0.0230085832403671, 0.0509842648495735, 0.0674163682329703, 0.0776298692537992,
0.0845763483455989, 0.0898956331467927, 0.0960783984315815, 0.0981374793725492],
        '250': [250, 0.00135394709228181, 0.00336084540489899, 0.00444187864153149,
0.00572264179835361, 0.00769140460909738, 0.011271757062258,
0.0232564240121588, 0.0512742310299524, 0.0674040655759355, 0.0778059485881525,
0.0845685557194811, 0.0900789327685887, 0.0965728744630847, 0.0985924133966425],
        '500': [500, 0.00133278513978836, 0.0033785281442177, 0.00443291182155661,
0.00566730197704589, 0.00759119298969563, 0.011106158112734,
0.0229299610528671, 0.0512804516178702, 0.0677306565298591, 0.0780913392144301,
0.0847229710024254, 0.0898507154096893, 0.0960716024468816, 0.0979756395318706],
        '750': [750, 0.00127271754613132, 0.00334326253946835, 0.0044376058858092,
0.00567836549601221, 0.00765416488538404, 0.011082843605444,
0.0233006332222607, 0.0510644551700367, 0.0672474116720296, 0.0779029087903253,
0.0847543574346385, 0.090129611707222, 0.0959893874257297, 0.0982990865065379],
        '1000': [1000, 0.00142855861269069, 0.00337982782600337,
0.00448136029314251, 0.00574442680663723, 0.00770970877283258,
0.0111369411954928, 0.0231006399903615, 0.0512968649949992, 0.0678321175431207,
0.0782466142059317, 0.0847527233328217, 0.0899530004868225, 0.0961991259484588,
0.0985702560735901],
        '1250': [1250, 0.0012216250863014, 0.00335627004382164,
0.00436451873181917, 0.00562010410578463, 0.00754792233150499,
0.0110624813220342, 0.023034471259804, 0.0511554847291635, 0.0675896285434318,
0.078123386919052, 0.0849540685548403, 0.0903162236226654, 0.0959418515751104,
0.0980227647050528],
        '2500': [2500, 0.0013993685161291, 0.00328278081675422,
0.00431275840777267, 0.00561088622839673, 0.00756357885754346,
0.0109304449056177, 0.022806018819766, 0.0510722515356924, 0.0673052939402961,
0.0780119775067037, 0.0853537397114059, 0.0904679666300433, 0.0963781569645721,
0.0984251958126395],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# egc_hurst_qtab <- egc_quantile_table("hurst")
egc_hurst_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 10 ** -4, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8,
                     0.9,
                     0.95, 0.975, 0.99, 0.999, 0.9999],
        '25': [25,
-9.63218513210034, -4.80494350822017, -3.7671534658725, -2.85277130883269,
-1.83692531355463, -0.752569171726627, 0.604000750059967, 1.35768448858258,
1.81882122386621, 2.33272216406319, 2.91221210511881, 3.82876761201107,
7.01862426321936, 9.1398433953275],
        '50': [50, -1.9050924494951,
-0.852513667442171, -0.570085633331304, -0.313161675091895, -0.0383113213022736,
0.285313459696212, 0.74243741102064, 0.974872661817781, 1.04321185041397,
1.08277705983682, 1.10858767296937, 1.13290665273223, 1.16840006853446,
1.19703629775581],
        '100': [100, -0.542341332261871, -0.000460799096853752,
0.138306958496461, 0.264338161116628, 0.397567270435087, 0.558043818507171,
0.812233608011752, 0.955234695370254, 1.00004004939875, 1.02613047521966,
1.04325821078398, 1.05862619690549, 1.07567022398738, 1.08150747353608],
        '250': [250, 0.0364467517721989, 0.36709342279675, 0.455844295660924,
0.529898175109903, 0.611772154136742, 0.706905728013722, 0.863899663533525,
0.961768656905356, 0.99514837605235, 1.01624853292206, 1.02924512646648,
1.04045633743049, 1.05451260940423, 1.05965663724796],
        '500': [500,
0.326156118737103, 0.543642758431625, 0.607657260251456, 0.661018050941538,
0.719786876344008, 0.787840979044478, 0.898654647603, 0.969857607372171,
0.994498525078497, 1.00926184204734, 1.01925987211215, 1.02802788622387,
1.03855548305486, 1.04346392193489],
        '750': [750, 0.379205027306303,
0.585209203892, 0.64438304015065, 0.694344443824018, 0.746016024794462,
0.80630119440741, 0.906517472215935, 0.972978852837563, 0.995022730726075,
1.008816347688, 1.01855284526596, 1.02637153691943, 1.03610859046259,
1.04086016968425],
        '1000': [1000, 0.426771191683228, 0.641826688088124,
0.693710039681907, 0.734606999492946, 0.780993517987521, 0.833751313940063,
0.919984211407626, 0.977806741856446, 0.997738239061993, 1.01057452818018,
1.01900471092174, 1.02607821865229, 1.03519697770899, 1.03896606650759],
        '1250': [1250, 0.575494248671508, 0.735200890947932, 0.776637758839186,
0.807213631869289, 0.843026054376433, 0.881278523914099, 0.94235670701181,
0.983235898640339, 0.998355191302432, 1.00808253560767, 1.01501108261563,
1.02068424207231, 1.02828229575173, 1.03126979783162],
        '2500': [2500,
0.789099270452659, 0.881814408383096, 0.900764494668801, 0.915952511467294,
0.931816490422547, 0.948733872894772, 0.974297337060417, 0.99181834977911,
0.998538124195272, 1.00277391299618, 1.00563617512021, 1.00833897419218,
1.01204733049964, 1.01345263549346],
    },
    index=['', '0.01%', '1%', '2.5%', '5%', '10%', '20%',
           '50%', '80%', '90%', '95%', '97.5%', '99%', '99.9%', '99.99%'],
    columns=['quantile', '25', '50', '100', '250', '500', '750', '1000', '1250',
             '2500']
)


# The following table was generated using the call
# rho_bias_qtab <- generate_rho_bias_table()
rho_bias_qtab = pd.DataFrame(
    {
        'quantile': [np.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
        '25': [25, 0.00579626402932596, 0.0890590962172455, 0.171824874686124,
0.251044677515644, 0.330764203025436, 0.405704835424967, 0.475061717710669,
0.541808016810147, 0.571735804493515, 0.60565221599304, 0.612682474856749,
0.619987042965707, 0.625029123229028, 0.632288580912326, 0.635008192179664,
0.646238773757011, 0.651437052422971, 0.657927899022773, 0.667817263227693,
0.672961524124665],
        '50': [50, 0.0541200753802938, 0.144640150821975,
0.235892360611822, 0.325879291413036, 0.41680267199482, 0.50415885702723,
0.588396212791385, 0.6740485464077, 0.71196928363215, 0.75104316423088,
0.760979456066076, 0.766067337337698, 0.771364755676878, 0.779816398200369,
0.785651205150849, 0.794421953991002, 0.803023539408362, 0.811087774056031,
0.81967359452155, 0.827239823674534],
        '100': [100, 0.0763382548072746,
0.171563568987254, 0.266425445954362, 0.363775028318011, 0.459018061866415,
0.551919940061675, 0.645037307308082, 0.739174729313482, 0.782816392968272,
0.827396377481796, 0.835864190196367, 0.844726057133227, 0.852873925321447,
0.861127936107442, 0.869938168307618, 0.877156691844852, 0.884687497867436,
0.894722344757547, 0.902689193366767, 0.911350711059478],
        '200': [200,
0.087114208110281, 0.186016801663539, 0.283912989935641, 0.382085135535338,
0.478690277940978, 0.576395103301607, 0.672975898263338, 0.77073179314775,
0.818743839975534, 0.865734334714667, 0.873894552241051, 0.883964126784713,
0.89339587111815, 0.901890231334893, 0.911357373612709, 0.920054300498443,
0.928996558707903, 0.937506720571201, 0.945549386762055, 0.955551294268134],
        '400': [400, 0.0940012978016627, 0.192671732277283, 0.291295089562392,
0.391330078301239, 0.489821049976308, 0.587921411146489, 0.686678482651164,
0.784917921704036, 0.834510089809465, 0.883079850902787, 0.893384611054417,
0.902728346204853, 0.912437644586801, 0.922288368568395, 0.9313875631715,
0.940933255717854, 0.950449393241092, 0.959701555797522, 0.968454005816006,
0.977142238885091],
        '800': [800, 0.0968384455762118, 0.196074149282283,
0.295709004626325, 0.395445405898641, 0.495164855910322, 0.595000938146491,
0.69404623598311, 0.792081225218621, 0.841700021015024, 0.891709672415359,
0.901715294627787, 0.911789232267656, 0.92120837824092, 0.93143708397091,
0.94120355072721, 0.9511699215617, 0.960582362268785, 0.97036408365614,
0.979744639399311, 0.988550300982111],
        '1200': [1200, 0.0983826928642641,
0.197569144039095, 0.297611810574128, 0.397267020262764, 0.496708586841542,
0.596078529037621, 0.695410540033741, 0.795106541068475, 0.844944485954799,
0.894523254206652, 0.90442222816685, 0.914444804451488, 0.924269988662038,
0.934192504230248, 0.944269402689644, 0.954136229383738, 0.963899750318928,
0.973770779185442, 0.983388098992297, 0.992459714269753],
        '1600': [1600,
0.0988539653135237, 0.198498983488058, 0.298226285609526, 0.398016646971277,
0.49693043926081, 0.596709186175247, 0.696654320053202, 0.796436355580036,
0.84632782271299, 0.895920019427508, 0.906039442133343, 0.915759135965388,
0.925843045012764, 0.935766422671986, 0.945614498189454, 0.95563174528189,
0.965547234932824, 0.97539219220646, 0.985112331636528, 0.994337351609841],
        '2000': [2000, 0.0984872136826968, 0.198566209126483, 0.298512558402136,
0.397919485467869, 0.498188663157584, 0.597854944882431, 0.697221012623204,
0.797359362655975, 0.84692400802886, 0.89681846757967, 0.906888349658203,
0.916805348446773, 0.926715861630496, 0.936635442905022, 0.946764917293779,
0.95645907431767, 0.966473528913937, 0.97638570289554, 0.986138462304169,
0.995504007250361],
    },
    index=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.85',
           '0.9', '0.91', '0.92', '0.93', '0.94', '0.95', '0.96', '0.97',
           '0.98', '0.99', '1'],
    columns=['quantile', '25', '50', '100', '200', '400', '800', '1200', '1600',
             '2000']
)


# This table is generated through the following steps:
# cbtab <- coint_bias_table(sample_size=c(25,50,100,200,400,800,1200,1600,2000))
# rho_bias_ltab <<- cbtab_to_rhobtab(cbtab)

# rho_bias_ltab <- structure(list(
# n = c(25, 50, 100, 200, 400, 800, 1200, 1600, 2000),
# c0 = c(0.08060, 0.03249, 0.01528, 0.00929, 0.00409, 0.00318, 0.00155, 0.00131, 0.00097),
# c1 = c(1.32771, 1.15184, 1.06583, 1.02898, 1.01415, 1.00517, 1.00441, 1.00238, 1.00258)),
# .Names = c("n", "c0", "c1"),
# row.names = c(NA, 9L), class = "data.frame")


rho_bias_ltab = pd.DataFrame(
    {
        'n': [25, 50, 100, 200, 400, 800, 1200, 1600,
2000],
        'c0': [0.0814875102581547, 0.0348936698142133, 0.0191890654195977,
0.00827078048614661, 0.00349860867799339, 0.00201210403167159,
0.00174662084510353, 0.00132323010209433, 0.000815444688383991],
        'c1': [1.22568817222908, 1.10252163308562, 1.0454757586257,
1.02296381021271, 1.01182912101022, 1.00533407590582, 1.00392839598619,
1.00300066638024, 1.00253862516874]
    },
    columns=['n', 'c0', 'c1']
)