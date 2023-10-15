#include "testref_resample.h"

std::vector<arma::fmat> get_testref_resample() {
    std::vector<arma::fmat> branches;
    arma::fmat current_branch;
	current_branch = {{277.0,280.0,285.3639610306789,291.02943725152284,293.0,291.0,},
	                  {0.0,8.757359312880714,14.636038969321072,19.0,11.798989873223332,2.6274169979695223,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{298.0,300.12132034355966,302.9497474683058,305.0,307.0,309.0,311.4350288425444,313.0,314.2634559672906,317.0,320.0,325.0,323.0,321.0,319.0,317.0,315.0,312.0,310.0,307.0,305.0,},
	                  {0.0,9.121320343559642,17.949747468305834,27.100505063388333,36.27207793864214,45.44365081389596,54.435028842544405,63.78679656440357,73.2634559672906,82.12994231491119,90.88730162779191,90.35533905932738,81.18376618407356,72.01219330881975,62.840620433565945,53.66904755831213,44.49747468305832,35.740115370177605,26.568542494923797,17.811183182043084,8.639610306789274,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{309.0,311.0,313.0,315.0,317.4852813742386,320.0,322.0,325.0,327.0,331.301515190165,332.0,330.0,328.0,326.0,324.0,322.0,320.0,317.0,},
	                  {0.0,9.17157287525381,18.34314575050762,27.51471862576143,36.48528137423857,45.44365081389596,54.615223689149765,63.37258300203048,72.54415587728428,80.0,71.5269119345812,62.35533905932738,53.18376618407357,44.01219330881976,34.840620433565945,25.669047558312137,16.49747468305833,7.740115370177612,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{321.0,326.0,},
	                  {0.0,3.2426406871192857,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{334.0,335.4142135623731,337.0,338.9497474683058,340.0,341.77817459305203,343.0,348.0208152801713,353.0,352.0,351.0,350.0,349.7989898732233,355.0,353.0,},
	                  {0.0,9.414213562373094,18.757359312880716,27.949747468305834,37.51471862576143,46.77817459305202,56.27207793864215,59.020815280171306,60.21320343559643,50.62741699796952,41.04163056034261,31.45584412271571,22.20101012677667,16.698484809834994,7.526911934581186,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{338.0,340.0,343.65685424949237,345.0,344.0,},
	                  {0.0,9.17157287525381,17.65685424949238,9.727922061357855,0.142135623730951,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = arma::fmat(2, 1);
	current_branch(0, 0) = 385.0;
	current_branch(1, 0) = 0.0;
	branches.emplace_back(std::move(current_branch));
	current_branch = {{412.0,411.0,},
	                  {0.0,9.585786437626904,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{417.0,417.0,417.0,413.1715728752538,412.0,412.0,412.0,411.75735931288074,406.0,403.0,401.0,400.0,404.55634918610406,405.0,409.09188309203677,},
	                  {0.0,10.0,20.0,27.82842712474619,37.34314575050762,47.34314575050762,57.34314575050762,67.24264068711929,62.89949493661166,54.14213562373095,44.970562748477136,35.384776310850235,28.443650813895953,18.62741699796952,13.091883092036785,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{419.0,423.0,},
	                  {0.0,4.828427124746192,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{444.0,440.0,438.0,434.0,433.0,428.0,427.0,423.6152236891498,423.0,424.5060966544099,432.74873734152914,439.2842712474619,440.0,441.0,444.0,445.0,446.3553390593274,450.59797974644664,452.7193000900063,456.0,},
	                  {0.0,8.34314575050762,17.51471862576143,25.85786437626905,35.44365081389596,42.78679656440357,52.37258300203048,60.384776310850235,70.12994231491119,79.50609665440987,76.25126265847084,74.7157287525381,65.01219330881976,55.42640687119285,46.669047558312144,37.083261120685236,27.644660940672626,19.402020253553342,10.2806999099937,1.639610306789283,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{475.0,472.0,469.0,466.0,462.9791847198287,460.85786437626905,458.02943725152284,455.0,452.0,451.0,451.0,457.57716446627535,466.1126983722081,468.0,469.0,471.0,473.1837661840736,476.0,478.84062043356596,481.0,484.0,486.0,},
	                  {0.0,8.757359312880714,17.51471862576143,26.27207793864214,35.020815280171306,44.14213562373095,52.97056274847714,61.7157287525381,70.4730880654188,80.05887450304571,90.05887450304571,90.42283553372465,88.88730162779191,79.66904755831214,70.08326112068524,60.91168824543143,51.81623381592644,42.9827560572969,34.159379566434055,25.05382386916238,16.296464556281663,7.124891681027854,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{505.0,502.87867965644034,500.0,497.9289321881345,495.0,492.0,489.0,485.90811690796323,483.0,481.0,479.5441558772843,479.0,485.0,489.94112549695427,496.0,499.0,501.0,504.0,507.0,509.2045814642449,512.0,515.0,517.6898628384835,520.0,},
	                  {0.0,9.121320343559642,17.928932188134524,27.071067811865476,35.85786437626905,44.615223689149765,53.37258300203048,62.09188309203678,70.88730162779191,80.05887450304571,89.4558441227157,99.23044737829953,97.01219330881976,89.05887450304571,83.32590180780451,74.5685424949238,65.39696961966999,56.63961030678928,47.88225099390856,38.795418535755125,29.95331880577404,21.195959492893326,12.310137161516558,3.267027304758802,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{536.0,533.8786796564403,531.0,528.221825406948,525.3933982822018,523.0,520.0,517.0,514.0796897832171,512.0,510.0,508.0,507.0,506.0,506.40559159102156,511.0,512.6482322781409,514.0,516.8908729652601,},
	                  {0.0,9.121320343559642,17.928932188134524,26.77817459305202,35.60660171779821,44.615223689149765,53.37258300203048,62.12994231491119,70.92031021678297,80.05887450304571,89.23044737829953,98.40202025355333,107.98780669118024,117.57359312880715,127.40559159102155,121.66904755831214,112.35176772185918,102.91168824543142,94.10912703473988,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{550.0,549.0,547.0,544.3431457505076,542.0,539.0,536.0,533.7365440327094,531.0,528.0,525.9583694396574,523.1299423149112,},
	                  {0.0,9.585786437626904,18.757359312880716,27.65685424949238,36.68629150101524,45.44365081389596,54.201010126776666,63.26345596729059,72.12994231491119,80.88730162779191,90.04163056034261,98.87005768508881,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{567.0,564.8786796564403,562.0,559.221825406948,557.0,554.0,551.0,548.6152236891497,545.7867965644036,543.0,540.1299423149112,538.0,536.0,535.0,532.3517677218591,},
	                  {0.0,9.121320343559642,17.928932188134524,26.77817459305202,35.85786437626905,44.615223689149765,53.37258300203048,62.384776310850235,71.21320343559643,80.05887450304571,88.87005768508881,97.98780669118024,107.15937956643405,116.74516600406096,125.64823227814082,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{583.0,581.0,579.0,576.0,573.8076118445749,571.0,568.0,565.3223304703363,562.4939033455901,560.0,557.0,554.7157287525381,552.0,549.0,},
	                  {0.0,9.171572875253808,18.343145750507617,27.100505063388333,36.192388155425114,45.02943725152286,53.78679656440357,62.67766952966369,71.50609665440987,80.4730880654188,89.23044737829953,98.2842712474619,107.15937956643405,115.91673887931476,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{607.0,599.1715728752538,595.636038969321,593.0,590.0,587.857864376269,585.0,582.0,579.0,576.5441558772843,573.7157287525381,571.0,568.0588745030457,565.9375541594861,563.0,560.2806999099937,554.0,551.0,549.0,556.4472221513641,563.5613998199874,568.9325035256027,573.8822509939085,577.4177848998413,580.2462120245875,583.0746391493337,586.0,589.0,591.5599205235723,594.3883476483185,597.0,600.0,602.873629022557,605.7020561473032,608.0,611.0,613.4802307403552,},
	                  {0.0,1.8284271247461903,10.363961030678928,19.27207793864214,28.029437251522857,37.14213562373095,45.95836943965738,54.7157287525381,63.47308806541881,72.4558441227157,81.2842712474619,90.15937956643405,98.94112549695429,108.06244584051392,116.84567106744929,125.7193000900063,131.9461761308376,140.70353544371835,149.87510831897214,149.55277784863586,146.0,139.06749647439727,131.11774900609143,122.5822151001587,113.75378797541251,104.92536085066632,96.13708498984761,87.3797256769669,78.44007947642775,69.61165235168156,60.69343417595165,51.936074863070935,43.12637097744299,34.2979438526968,25.249783362055695,16.492424049174982,7.519769259644778,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{659.0,655.0,651.221825406948,648.0,644.1507575950825,641.0,638.0,634.9583694396574,631.4228355337247,628.0,625.0,622.0,619.0,615.8664863476206,612.3309524416878,609.0,605.966991411009,603.0,599.60303038033,596.7746032555838,593.9461761308377,591.0,588.0,585.460894756599,582.6324676318528,580.0,577.0,574.1471862576143,572.0,569.0,566.3690116645623,563.5405845398161,561.0,558.0,555.0553031655775,552.9339828220179,550.0,547.0,544.0,541.6202743230331,539.0,536.6705268547273,534.0,531.7207793864214,529.0,526.7710319181156,524.0,521.8212844498098,519.0,516.8715369815039,514.0,511.9217895131981,509.0,506.26493526370575,504.0,501.3151877953999,499.0,},
	                  {0.0,8.34314575050762,16.778174593052025,25.443650813895957,33.8492424049175,42.5441558772843,51.301515190165006,60.04163056034262,68.57716446627536,77.15937956643405,85.91673887931478,94.67409819219549,103.4314575050762,112.1335136523794,120.66904755831214,129.28932188134524,138.03300858899107,146.8040405071067,155.39696961967,164.22539674441617,173.0538238691624,181.83347775862956,190.59083707151026,199.53910524340094,208.36753236814712,217.2771285725255,226.0344878854062,234.8528137423857,243.96342007354073,252.72077938642144,261.63098833543773,270.45941546018395,279.4070708874367,288.1644302003174,296.9446968344225,306.06601717798213,314.85072170133265,323.6080810142134,332.36544032709406,341.37972567696687,350.2943725152286,359.32947314527274,368.2233047033631,377.27922061357856,386.15223689149764,395.2289680818844,404.08116907963216,413.17871555019025,422.0101012677667,431.12846301849606,439.9390334559012,449.0782104868019,457.86796564403573,466.73506473629425,475.79689783217026,484.6848122046001,493.7258300203048,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{673.0,666.1715728752538,661.9289321881346,657.6862915010153,654.0,650.0,647.0,643.0,640.0,636.4730880654188,633.0,630.0,626.5735931288071,623.0380592228744,620.0,617.0,613.8456710674493,611.0,608.0,605.0,602.0,599.7035354437184,597.0,594.0,591.0,588.3898269447336,586.0,583.0,580.0,577.0761184457489,574.9547981021892,572.0,569.0,566.4695167279506,563.6410896032045,561.0,558.0,555.1558082289658,553.0,550.0,547.0,545.0,542.0,539.5994590428618,537.0,534.649711574556,532.0,529.0,527.0,524.0431098567577,522.0,519.093362388452,517.0,514.1436149201461,512.0,509.0,507.0,504.0,},
	                  {0.0,4.82842712474619,13.071067811865476,21.31370849898476,29.786796564403573,38.12994231491119,46.887301627791906,55.23044737829953,63.98780669118024,72.52691193458118,81.08831175456858,89.84567106744929,98.42640687119285,106.96194077712559,115.70353544371834,124.46089475659906,133.1543289325507,141.97561338236048,150.7329726952412,159.49033200812192,168.24769132100263,177.29646455628165,186.17662350913716,194.93398282201787,203.69134213489858,212.61017305526642,221.6202743230331,230.3776336359138,239.13499294879455,247.92388155425118,257.04520189781084,265.82128444980975,274.5786437626905,283.53048327204937,292.3589103967956,301.26493526370575,310.02229457658643,318.84419177103416,327.95122676472096,336.7085860776017,345.4659453904824,354.6375182657362,363.3948775786169,372.40054095713816,381.3238097667514,390.35028842544403,399.25274195488595,408.0101012677667,417.1816741430205,425.9568901432422,435.110606331155,443.9066376115481,453.0395385192895,461.8563850798539,470.96847070742405,479.7258300203048,488.89740289555857,497.6547622084393,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{691.0,688.0,684.0,680.3933982822018,676.857864376269,673.0,669.7867965644036,666.0,662.0086219713515,659.0,655.0,655.8456710674493,661.0,664.6690475583122,668.2045814642448,672.0,675.9827560572969,679.0,683.0,686.5893577750951,690.1248916810279,694.0,},
	                  {0.0,8.757359312880714,17.100505063388333,25.606601717798213,34.14213562373095,42.54415587728428,51.21320343559643,59.64466094067262,67.99137802864846,76.74516600406096,85.08831175456858,92.0,84.8111831820431,76.33095244168787,67.79541853575513,59.36753236814713,51.017243942703104,42.2670273047588,33.92388155425118,25.41064222490489,16.875108318972153,8.480230740355225,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{642.0,633.7573593128807,628.0,624.9791847198287,622.0,619.0,616.0,613.0,610.1299423149112,608.0,605.0,602.0,599.523340597113,597.0,594.0,591.0380592228744,588.2096320981282,582.5527778486359,580.9827560572969,588.0,587.0,591.4680374315354,595.0,598.0,601.3675323681472,605.0,609.0,612.0,615.0,617.6309883354377,620.4594154601839,623.0,626.1162697096763,629.0,632.0,635.0,638.0,641.0,643.79393923934,644.0847404171004,},
	                  {3.0,6.757359312880714,10.272077938642145,19.02081528017131,27.786796564403573,36.54415587728429,45.301515190165006,54.058874503045715,62.8700576850888,71.98780669118024,80.74516600406096,89.50252531694167,98.47665940288702,107.4314575050762,116.1888168179569,124.96194077712559,133.79036790187178,141.44722215136417,149.9827560572969,149.3187591328681,158.904545570495,159.53196256846454,150.99494936611666,142.23759005323595,133.63246763185288,125.13708498984761,116.79393923933999,108.03657992645927,99.27922061357856,90.36901166456228,81.54058453981608,72.59292911256333,63.88373029032371,55.07821048680189,46.32085117392118,37.56349186104046,28.80613254815975,20.048773235279036,11.206060760660021,2.084740417100379,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{378.0,375.0,371.0,372.0,372.0,373.0,374.0,376.0,380.0,380.0,381.8492424049175,383.0,383.0,382.0,381.32233047033634,378.0,},
	                  {15.0,12.242640687119286,17.51471862576143,27.100505063388333,37.10050506338833,46.68629150101523,56.27207793864214,65.44365081389596,61.38477631085024,51.38477631085024,42.1507575950825,32.627416997969526,22.627416997969526,13.041630560342618,3.3223304703363135,10.715728752538096,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{274.0,273.0,274.0,272.0,275.19238815542514,279.6152236891498,281.0,283.0,281.4939033455901,279.37258300203047,277.0,},
	                  {14.0,23.585786437626904,33.17157287525381,41.51471862576143,50.19238815542512,58.0,50.21320343559643,41.8700576850888,32.49390334559012,23.372583002030478,14.355339059327374,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{184.0,184.53553390593274,188.07106781186548,195.0,192.0,188.3223304703363,},
	                  {16.0,23.535533905932738,32.071067811865476,34.55634918610404,25.798989873223334,17.322330470336315,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{395.0,393.0,392.0,390.75735931288074,387.0,384.0,384.0,380.0,377.0,375.4939033455901,373.0,371.0,372.16295090390224,374.0,379.1126983722081,380.0,380.52691193458116,382.6482322781408,385.0,387.59797974644664,390.42640687119285,392.0,394.6690475583121,398.91168824543144,399.0,399.0,399.0,398.0,},
	                  {52.0,61.17157287525381,70.75735931288071,80.24264068711928,88.68629150101523,97.44365081389596,106.61522368914976,114.37258300203048,123.12994231491119,132.50609665440987,141.4730880654188,150.64466094067262,160.16295090390227,169.40202025355333,175.8873016277919,166.25483399593904,156.4730880654188,147.35176772185918,138.3259018078045,129.40202025355333,120.57359312880715,111.22539674441619,102.33095244168787,94.08831175456858,84.12489168102785,74.12489168102785,64.12489168102785,54.53910524340095,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{271.0,272.5355339059327,276.6360389693211,272.14213562373095,},
	                  {57.0,65.53553390593274,63.63603896932107,56.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{362.0,366.0,364.46446609406723,363.0,358.0,359.0,},
	                  {84.0,80.82842712474618,71.46446609406726,62.071067811865476,59.27207793864214,68.85786437626905,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{352.0,352.0,353.53553390593277,355.0,357.77817459305203,361.85786437626905,362.0,361.0,359.0,357.0,353.2842712474619,},
	                  {68.0,77.17157287525382,86.53553390593274,95.92893218813452,104.77817459305203,112.0,102.97056274847714,93.38477631085024,84.21320343559643,75.04163056034261,67.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{377.0,378.24264068711926,},
	                  {70.0,69.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{268.0,269.53553390593277,274.1005050633883,274.1005050633883,270.5649711574556,},
	                  {71.0,79.53553390593274,87.0,79.10050506338834,70.5649711574556,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{307.0,307.12132034355966,310.0,312.0710678118655,316.85786437626905,317.0,315.0,312.7365440327094,308.2842712474619,},
	                  {76.0,85.12132034355965,93.92893218813452,103.07106781186548,110.0,100.97056274847714,91.79898987322333,82.7365440327094,75.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{344.0,344.0,346.0,348.0,351.5147186257614,350.0,348.0,344.85786437626905,},
	                  {81.0,90.17157287525382,99.34314575050762,108.51471862576143,106.51471862576143,97.14213562373095,87.97056274847714,80.14213562373095,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = arma::fmat(2, 1);
	current_branch(0, 0) = 361.0;
	current_branch(1, 0) = 80.0;
	branches.emplace_back(std::move(current_branch));
	current_branch = {{263.0,264.0,261.0,264.0,268.14213562373095,272.32233047033634,275.0,272.0,269.0,},
	                  {82.0,90.75735931288071,98.10050506338834,106.85786437626905,115.14213562373095,112.32233047033631,103.698484809835,94.94112549695429,86.18376618407356,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{861.0,854.0502525316941,846.9791847198287,840.6152236891497,836.0,844.7487373415291,851.8198051533947,859.1837661840735,865.5477272147525,867.2964645562816,},
	                  {97.0,102.94974746830583,110.0208152801713,117.38477631085023,125.4730880654188,126.25126265847084,119.18019484660536,112.81623381592644,105.4522727852475,98.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{395.0,397.12132034355966,400.0,402.0710678118655,404.8994949366117,407.0208152801713,409.8492424049175,412.0,415.0,417.0,420.0,422.0,425.0,427.0,430.0,432.0,435.0,437.42640687119285,440.0,442.37615433949867,445.0,447.32590180780454,450.0,452.9827560572969,455.0,457.9325035256027,460.0,462.88225099390854,465.0,467.8319984622144,470.0,472.78174593052023,475.0,477.73149339882605,480.0,483.0,485.0,488.0,490.0,493.0,495.0,497.0,500.0,502.0,503.0,498.98423535371205,496.0,494.0,491.0,488.3776336359138,486.0,483.427886167608,481.0,479.0,476.3568183557425,474.0,471.0,469.0,466.45732341913083,464.0,461.507575950825,459.0,456.5578284825192,454.0,452.0,449.0,447.0,444.0,442.0,439.0,437.0,434.0,432.0,429.0,427.0,424.030916547938,422.0,419.08116907963216,417.0,414.13142161132635,412.0,409.18167414302053,407.0,404.9390334559012,402.0,399.28217920640884,397.0,395.0,392.0,390.0,},
	                  {499.0,489.87867965644034,481.0710678118655,471.9289321881345,463.1005050633883,453.9791847198287,445.1507575950825,436.04163056034264,427.2842712474619,418.1126983722081,409.3553390593274,400.1837661840736,391.42640687119285,382.25483399593907,373.49747468305833,364.32590180780454,355.5685424949238,346.57359312880715,337.6396103067893,328.62384566050133,319.71067811865476,310.67409819219546,301.78174593052023,293.0172439427031,283.8528137423857,275.0674964743973,265.9238815542512,257.11774900609146,247.99494936611666,239.1680015377856,230.06601717798213,221.21825406947977,212.1370849898476,203.26850660117395,194.20815280171308,185.45079348883237,176.27922061357856,167.52186130069785,158.35028842544403,149.59292911256333,140.4213562373095,131.2497833620557,122.49242404917499,113.32085117392117,103.73506473629426,98.01576464628796,106.77965388946717,115.95122676472097,124.7085860776017,133.62236636408616,142.63751826573622,151.572113832392,160.56645045387074,169.73802332912456,178.64318164425748,187.66695551725908,196.4243148301398,205.5958877053936,214.54267658086914,223.52481989352813,232.49242404917499,241.45375208166266,250.4421715174808,259.38268426979715,268.554257145051,277.3116164579317,286.4831893331855,295.2405486460662,304.41212152132005,313.1694808342007,322.34105370945457,331.09841302233525,340.2699858975891,349.0273452104698,358.1989180857236,366.969083452062,376.12785027385814,384.91883092036784,394.05678246199267,402.86857838867365,411.9857146501272,420.81832585697947,429.9146468382617,439.0609665440988,447.84357902639624,456.71782079359116,465.77251121453077,474.94408408978455,483.7014434026653,492.8730162779191,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{530.0,530.0,528.8786796564403,525.3431457505076,523.9289321881346,522.0,523.4436508138959,529.0,535.1299423149112,540.0,542.0,},
	                  {128.0,118.0,110.12132034355965,118.65685424949238,128.07106781186548,137.27207793864216,145.0,138.21320343559643,136.0,129.1126983722081,119.94112549695429,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{260.0,262.0,267.77817459305203,270.0,267.0,261.4558441227157,260.0,},
	                  {126.0,134.34314575050763,139.22182540694797,130.14213562373095,121.38477631085023,114.0,122.301515190165,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{253.0,254.7573593128807,263.3639610306789,262.0,264.0,260.02943725152284,254.0,253.54415587728428,254.0,},
	                  {162.0,168.0,164.63603896932108,155.72792206135784,147.38477631085024,139.02943725152286,139.8873016277919,147.45584412271572,155.98780669118025,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{58.0,61.242640687119284,67.60660171779821,73.97056274847714,81.88730162779191,80.83704909609774,74.47308806541882,67.10912703473988,60.03805922287441,},
	                  {143.0,150.2426406871193,157.6066017177982,164.97056274847714,170.0,162.83704909609773,155.4730880654188,149.10912703473988,142.0380592228744,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{248.0,249.0,247.0,246.0,252.55634918610406,256.0,258.0,255.54415587728428,},
	                  {172.0,180.7573593128807,189.10050506338834,197.02943725152286,197.44365081389594,188.8700576850888,180.5269119345812,171.54415587728428,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{64.0,68.36396103067892,76.0208152801713,72.32233047033631,64.95836943965739,},
	                  {174.0,181.36396103067892,186.97918471982868,179.3223304703363,173.0416305603426,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{356.0,354.0,353.87867965644034,351.0502525316942,348.22182540694797,347.0,344.0,341.85786437626905,339.0,337.0,334.7867965644036,336.04163056034264,342.99137802864846,344.0,346.0,348.6482322781408,351.0,353.0,356.42640687119285,359.0,361.0,363.0,362.0,360.0,},
	                  {193.0,202.17157287525382,212.12132034355963,220.94974746830584,229.77817459305203,239.27207793864216,248.02943725152286,257.14213562373095,265.95836943965736,275.1299423149112,284.2132034355964,293.04163056034264,297.00862197135154,287.42640687119285,278.25483399593907,269.3517677218592,260.32590180780454,251.1543289325507,242.57359312880715,233.63961030678928,224.46803743153546,215.29646455628165,205.71067811865476,196.53910524340094,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{216.0,214.46446609406726,211.0,214.02081528017132,219.0,222.38477631085024,225.0,226.33452377915606,229.16295090390227,231.99137802864846,232.0,236.0,236.10912703473988,241.0,242.0,242.6238456605013,245.20458146424488,249.0,248.27564927611036,250.0,250.36038969321072,243.0,241.0,240.0,237.92536085066632,235.68272016354703,232.55992052357226,232.0,231.0,226.0,226.76240994676405,224.48023074035524,},
	                  {318.0,326.53553390593277,335.1005050633883,343.0208152801713,338.2132034355964,329.6152236891498,320.698484809835,312.6654762208439,303.83704909609776,295.00862197135154,285.84062043356596,278.32590180780454,269.1091270347399,261.81118318204307,252.22539674441617,243.6238456605013,235.79541853575512,227.36753236814712,218.72435072388964,210.8528137423857,202.36038969321072,206.83347775862953,215.17662350913716,223.93398282201787,233.07463914933368,241.31727983645297,249.55992052357226,258.13499294879455,266.0639251369291,273.9928573250636,281.2375900532359,288.48023074035524,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = arma::fmat(2, 1);
	current_branch(0, 0) = 246.0;
	current_branch(1, 0) = 217.0;
	branches.emplace_back(std::move(current_branch));
	current_branch = {{150.0,153.34314575050763,154.92893218813452,},
	                  {228.0,235.0,228.92893218813452,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{138.0,140.2426406871193,146.27207793864216,149.68629150101523,144.02943725152286,},
	                  {238.0,246.2426406871193,253.0,247.68629150101523,240.02943725152286,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = arma::fmat(2, 1);
	current_branch(0, 0) = 224.0;
	current_branch(1, 0) = 291.0;
	branches.emplace_back(std::move(current_branch));
	current_branch = {{220.0,220.2426406871193,},
	                  {300.0,308.24264068711926,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = arma::fmat(2, 1);
	current_branch(0, 0) = 219.0;
	current_branch(1, 0) = 310.0;
	branches.emplace_back(std::move(current_branch));
	current_branch = {{320.0,319.0,317.0,314.34314575050763,312.0,310.0,307.0,305.0,302.32233047033634,300.0,298.62741699796953,306.0,308.0,310.8198051533946,313.0,315.7695526217005,318.0,321.0,323.0,326.0,328.0,325.381204973382,},
	                  {331.0,340.5857864376269,349.75735931288074,358.65685424949237,367.68629150101526,376.85786437626905,385.6152236891498,394.7867965644036,403.67766952966366,412.7157287525381,421.62741699796953,423.1837661840736,414.01219330881975,405.1801948466054,396.0832611206852,387.2304473782995,378.1543289325507,369.39696961967,360.2253967444162,351.4680374315355,342.29646455628165,333.381204973382,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{635.0,638.0,},
	                  {367.0,369.65685424949237,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{612.0,612.2426406871193,615.0,},
	                  {369.0,377.24264068711926,371.8994949366117,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{614.0,615.6568542494924,624.4852813742385,619.142135623731,},
	                  {382.0,389.65685424949237,387.5147186257614,383.0,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{594.0,597.0,},
	                  {391.0,392.24264068711926,}};
	branches.emplace_back(std::move(current_branch));
	current_branch = {{601.0,595.0,602.363961030679,604.0,},
	                  {404.0,409.75735931288074,413.6360389693211,404.31370849898474,}};
	branches.emplace_back(std::move(current_branch));

    return branches;
}