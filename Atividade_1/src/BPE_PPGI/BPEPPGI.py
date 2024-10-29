import sys
from .utils.jsondataset import JSONDataset
import regex as re
import os
import joblib

# Code was implemented by following the video https://www.youtube.com/watch?v=zduSFxRajkE
class BPEBase():
    def __init__(self, vocabulary=None, pars=None, merges=None):
        if vocabulary == None:
            self.vocabulary = {idx: bytes([idx]) for idx in range(256)}
        else:
            self.vocabulary = vocabulary
            
        if pars == None:
            self.pairs = {}
        else:
            self.pairs = pars
            
        if merges == None:
            self.merges = {}
        else:
            self.merges = merges
        
    def tokenize(self, text):
        ids  = self.encode(text)
        tokens = [self.vocabulary[idx].decode("utf-8", errors="replace") for idx in ids]
        return tokens
    
    def encode(self, text):
        tokens = list(map(int, text.encode("utf-8")))
        while len(tokens) >= 2:
            stats = self.get_pair_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                return tokens
            idx = self.merges[pair]
            tokens = self.merge_pair(tokens, pair, idx)
            
        return tokens
    
    def decode(self, ids):
        tokens = b"".join(self.vocabulary[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def train(self, dataset, max_vocab_size, special_tokens_add_list=None):
        pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        max_merges = max_vocab_size - len(self.vocabulary)
        if max_merges < 0:
            max_merges = 0
                    
        tokens = []
        stats = {}
        for text in dataset:
            tks = re.findall(pat_str, text)
            lids = [list(map(int, it.encode("utf-8"))) for it in tks]
            for ids in lids:
                self.get_pair_stats(ids, stats)
            tokens.extend(lids)
            
        for i in range(max_merges):
            aux = []
            if len(stats) == 0:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            stats = {}
            for tks in tokens:
                new_ids = self.merge_pair(tks, pair, idx) # na lista de tokens, trocar o par pelo idx
                self.get_pair_stats(new_ids, stats)
                aux.append(new_ids)
            tokens = aux
            self.merges[pair] = idx
            self.pairs[idx] = pair
            self.vocabulary[idx] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]] # ao invés de salvar os pares, concatena os bytes para cada avanço e o decoder não precisa voltar recursivo
        
        return self
    
    def get_pair_stats(self, ids, counts=None):
        if counts == None:
            counts = {}
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_pair(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def __len__(self):
        return len(self.vocabulary)
        
    def save(self, dir):
        if not os.path.exists(dir):
            try:
                os.mkdir(dir)
            except:
                raise(IOError, "Error creating directory %s" % dir)
        
        with open(os.path.join(dir, 'vocabulary.dict'), 'bw') as v:
            joblib.dump(self.vocabulary, v)

        with open(os.path.join(dir, 'pars.dict'), 'bw') as p:
            joblib.dump(self.pairs, p)
            
        with open(os.path.join(dir, 'merges.dict'), 'bw') as m:
            joblib.dump(self.merges, m)
            
    def load(self, dir):
        if os.path.exists(dir):
            try:
                with open(os.path.join(dir, 'vocabulary.dict'), 'br') as v:
                    self.vocabulary = joblib.load(v)
            except:
                raise(IOError, "Error loading vocabulary in directory %s" % dir)
            
            try:
                with open(os.path.join(dir, 'pars.dict'), 'br') as p:
                    self.pairs = joblib.load(p)
            except:
                raise(IOError, "Error loading pairs in directory %s" % dir)
            
            try:
                with open(os.path.join(dir, 'merges.dict'), 'br') as m:
                    self.merges = joblib.load(m)
            except:
                raise(IOError, "Error loading merges in directory %s" % dir)

            
if __name__ == "__main__":
    dir = sys.argv[1]
    max_vocab_size = int(sys.argv[2])
    dataset = JSONDataset(dir)
    bpe = BPEBase()
    bpe.train(dataset, max_vocab_size=max_vocab_size)
    txt = "Teste de codificação."
    #print(list(map(int, txt.encode("utf-8"))))
    print(bpe.encode(txt))
    text2 = bpe.decode(bpe.encode(txt))
    print(text2 == txt)
    text = "{{Info/Município do Brasil | nome = Porteiras | foto = | leg_foto = | apelido = Conceição do Cariri | brasão = Brasão de Porteiras-CE.jpg | bandeira = Flag of Porteiras Municipality in the State of Ceará in Brazil.jpg | link_brasão = | link_bandeira = | link_hino = | aniversário = 25 de março | fundação = | emancipação = | gentílico = porteirense | lema = | prefeito = Fabio Pinheiro Cardoso | partido = PTB | fim_mandato = 2024 | mapa = Ceara Municip Porteiras.svg | latP = S | latG = 07 | latM = 32 | latS = 06 | lonP = O | lonG = 39 | lonM = 07 | lonS = 04 | estado = Ceará | mesorregião = Sul Cearense | data_mesorregião = IBGE/2008 | microrregião = Cariri | data_microrregião = IBGE/2008 | região_metropolitana = | vizinhos = Norte: Brejo Santo e Missão Velha; Sul: Jardim e Jati; Leste: Brejo Santo; Oeste: Missão Velha e Jardim | dist_capital = 530 | área = 217.570 | área_ref = | população = 17050 | data_pop = IBGE/2010 | altitude = 538 | clima = Tropical quente semiárido brando | sigla_clima = | idh = 0.622 | data_idh = PNUD/2010 | pib = mil | data_pib = IBGE/2008 | pib_per_capita = 3210.17 | data_pib_per_capita = IBGE/2008|thumb|90px | site_prefeitura = porteiras.ce.gov.br }} Porteiras é um município brasileiro localizado no sul do estado do Ceará, região Nordeste do país, a 520 km da capital, Fortaleza e 538 metros acima do nível do mar. Apesar de emancipada oficialmente em 25 de março de 1953, seu povoamento é um dos mais antigos da região, dentro da área habitada pela nação dos índios Cariris. Localizada no sopé da Chapada do Araripe, Porteiras é parte da Microrregião do Cariri, com uma população de aproximadamente 17.050 habitantes no ano de 2010 e um território estimado em 217,570 km² conforme estudo do Instituto Brasileiro de Geografia e Estatística (IBGE). == História == === Etimologia === O nome do distrito teria surgido devido a duas porteiras - chamadas de Porteira de Fora e Porteira de Dentro - que ficavam ao lado da Lagoa Ariosa. Na elevação onde hoje está localizado o centro da cidade, ficavam currais onde eram mantidos os bois. As porteiras delimitavam o terreno e controlavam os fluxos de entrada e saída de pessoas e do gado. Em 1920, a vila se chamou Conceição do Cariri (uma alusão à padroeira, Nossa Senhora da Conceição), mas voltou a se chamar Porteiras em 1938. === Primeiros habitantes === O município está dentro da área habitada pela nação dos índios Cariris. Contudo, as origens do povoamento não-indígena de Porteiras estão relacionadas ao município de Jardim, ao qual pertencia, já que, no século XVIII, o sítio Simão (hoje Distrito do Simão) era uma importante via de acesso para a cidade. Acredita-se que os primeiros povoadores não-indígenas do município foram atraídos para a região pela riqueza da terra fértil, própria para o desenvolvimento da agricultura e pela abundância de água que jorra do sopé serrano da Chapada do Araripe. Um deles foi o pernambucano José Antônio de Souza, capitão procedente do antigo município de Baixa-Verde, atual município de Triunfo. === \"Fogo das Guaribas\" e assassinato de Chico Chicote === Em fevereiro de 1927, Porteiras foi cenário de uma das passagens de Lampião e seu bando, no episódio sanguinário que ficaria conhecido como \"Tragédia das Guaribas\", ou \"Fogo nas Guaribas\", marco da história do cangaço e do coronelismo brasileiro. Chegando à vila de Porteiras, o cangaceiro teria alertado Maria Celina Grangeiro sobre a chegada do Tenente José Gonçalves Pereira ao Cariri em busca de seus inimigos, entre eles seu marido, o fazendeiro Antônio Grangeiro, e seu compadre Francisco Pereira de Lucena, o Chico Chicote, irmão do prefeito de Brejo Santo e conhecido por sua desobediência. Após 31 horas de troca de tiros entre o tenente e os capangas de Chico Chicote, Antônio Grangeiro, seu sobrinho e mais dois moradores foram raptados, torturados e assassinados. Um mausoléu em memória dos mortos foi erguido pelo filho de Grangeiro próximo ao Sítio Guaribas, entre os atuais territórios de Porteiras e Brejo Santo. ===Histórico administrativo e emancipação=== Criada como distrito, em denominações que datam de 1858, Porteiras foi elevada à categoria de vila em 17 de agosto de 1889, sendo desmembrada de Jardim. Em divisão administrativa de 1933, figurava como distrito pertencente ao município de Brejo dos Santos da Conceição (atual Brejo Santo) sob o nome de 'Conceição do Cariri'. Anos depois, por decreto estadual de 1938, voltou a se chamar Porteiras, e o município de Brejo dos Santos passou a se chamar Brejo Santo. Porteiras foi elevada à categoria de município com distrito único em novembro de 1951, quando foi finalmente desmembrada de Brejo Santo. Sua emancipação se consolidou em 25 de março de 1953 e é comemorada todos os anos, quando uma semana de festividades, eventos culturais e prestação de serviços é promovida. ==Geografia== Porteiras é parte da Microrregião do Cariri, fazendo limite com os municípios de Brejo Santo e Missão Velha ao norte e Jardim e Jati ao sul. Está localizada no bioma da caatinga, no semiárido brasileiro. Localizada ao sopé da Chapada do Araripe, está a uma altitude média de 538 metros acima do nível do mar. Sua vegetação inclui floresta caducifólia espinhosa, Floresta subcaducifólia tropical pluvial e Floresta subperenifólia tropical pluvio-nebular. === Clima === O clima porteirense é tropical quente semiárido brando. As chuvas se concentram de janeiro a abril. Variando entre clima seco e frio, as temperaturas alternam entre 12 e 28 graus. == Política == === Administração === Conforme modelo previsto pela Constituição Federal, que classifica o Brasil como uma república federativa presidencialista. A administração no município de Porteiras, portanto, ocorre dos poderes executivo e legislativo. O poder executivo é representado pela prefeitura, cuja sede está localizada na Rua Mestre Zuca, no centro da cidade. O primeiro prefeito de Porteiras foi José Aristarco Cardoso, e e atual é o empresário e político Fábio Pinheiro Cardoso, filiado ao Partido Trabalhista Brasileiro (PTB). Fábio foi eleito nas eleições municipais de 2016 no 1º turno, com 69,37% dos votos válidos, tendo como vice-prefeito o médico Aníbal Tavares de Caldas, também filiado ao PTB.Sob o guarda-chuva da prefeitura de Porteiras estão as secretarias de Administração, Agricultura e Meio Ambiente, Educação, Cultura e Desporto, Finanças, Obras, Saúde e Assistência Social. O centro do poder legislativo local, por sua vez, é o paço Gerônimo Manoel do Nascimento, sede da Câmara Municipal de Porteiras. Juntamente do prefeito, onze vereadores são eleitos a cada quatro anos, conforme previsto pela Constituição. O atual presidente da Câmara Municipal é o vereador Raimundo Nogueira de Lima, e o primeiro presidente da Câmara Municipal foi Enoque Tavares Miranda. == Demografia e indicadores == A população estimada do município de Porteiras no censo do IBGE em 2019 era de 14.996 habitantes, apresentando uma densidade densidade demográfica de 69,22 habitantes/km². Este número é menor que os 15.061 habitantes do último censo (2010). Uma amostra de 2010 indicou que 70.2% da população de Porteiras se autodenomina parda; 24.2% branca; 4,33% preta e 1.15% amarela. A taxa de escolarização de crianças entre 6 e 14 anos de idade é de 97,6%. === Religião === Porteiras é uma cidade majoritariamente cristã. Dividida sobretudo entre duas denominações. 93% da população se identifica como católica, seguidos pelos protestantes, que somam 5,8%. 1% dos porteirenses seguem outras denominações ou não possuem religião. A maior igreja do município é a Paróquia de Nossa Senhora da Conceição, que realiza anualmente uma tradicional festa de coroação, sempre no dia 31 de maio. O templo, localizado na rua Luís Grangeiro, um dos pontos mais elevados da zona urbana, foi construído em mutirão organizado pelo Padre Ibiapina em uma de suas passagens pela região. Na ocasião, o pároco também foi responsável por levar água à localizade através da construção de uma levada, do pé da serra à sede. A primeira Coroação de Nossa Senhora ocorreu em 1934, organizada pela professora Maria do Carmo Simplício, que deu nome, em 2005, à Biblioteca Pública Municipal. Festividades religiosas no município também incluem o Dia de Nossa Senhora da Conceição, padroeira da cidade. As celebrações são realizadas todo dia 8 de dezembro. Além da igreja católica, há também templos de outras denominações cristãs na cidade, como a Assembleia de Deus, o Salão do Reino das Testemunhas de Jeová e a Congregação Cristã no Brasil. == Cultura e lazer == === Comunidade quilombola === O Sítio Vassourinha, na zona rural de Porteiras, abriga o Quilombo dos Souza, certificado pela Fundação Cultural Palmares em 2005, sendo o terceiro quilombo do Ceará a receber este reconhecimento. A comunidade conta com 45 famílias cadastradas e autorreconhecidas, e tem como uma de suas lideranças culturais a Mestra Maria Josefa da Conceição, bisneta de Raimundo Valentim de Souza, que fugiu da escravidão nos engenhos de Pernambuco. Mestra Maria de Tiê, como é conhecida, foi nomeada Tesouro Vivo da Cultura pelo Governo do Estado do Ceará, mantendo vivas as tradições da família, como o reisado, confecções de bonecas negras, a dança maneiro-pau e os grupos de coco, que reúnem crianças e idosos. === Mídia e comunicações === Os principais meios de comunicação são o rádio, a internet, a televisão e as telefonias fixa e móvel. O código de área (DDD) de Porteiras é o 88, assim como em todas as cidades do centro-sul cearense. Na telefonia fixa, Porteiras estava inclusa na área de abrangência da estatal Teleceará, que, em julho de 1998, com a aprovação da Lei Geral de Telecomunicações no governo de Fernando Henrique Cardoso, foi privatizada, vendida para a Telemar, e, em 2007, passou a se chamar Oi. Serviços de telefonia móvel são oferecidos pela operadora Vivo desde a segunda metade da década de 2000, que, gradualmente, passou a oferecer serviços de telefonia 2G, 3G e 4G. Desde 1996, o município possui uma estação comunitária de rádio, a Conceição do Cariri FM, que alcança grande parte do município na frequência 104,9 MHz, além de partes de Brejo Santo, Jati, Jardim e Penaforte. Recebe também, em algumas localidades, o áudio da Rádio AM Sul Cearense de Brejo Santo, na frequência 820MHz. A televisão aberta recebe os sinais da TV Verdes Mares Cariri, afiliada da Rede Globo em Juazeiro do Norte, e da TV Diário, pelo canal 14 . Grande parte dos moradores recorre à TV via satélite por meio de antena parabólica, tendo acesso à maioria dos canais abertos do território nacional, e às operadoras Claro TV e Sky, de TV por assinatura. === Esporte === Equipamentos públicos para o fomento da atividade esportiva são de responsabilidade da Secretaria Municipal de Cultura e Desporto, como o Estádio Municipal Alboíno Miranda Tavares, o Mirandão, localizado no bairro Sol Nascente, o Ginásio Poliesportivo O Teixeirão e o Pólo de Lazer Luiz Caldas Campos. Os principais times de futebol da cidade são a Seleção Porteirense e o Campo Santo, do bairro Alto do Cemitério, os autodenominados \"Cangalhas de Porteiras\", atuais campeões do Campeonato Intermunicipal do Cariri. Porteiras é a cidade natal do futebolista Ronaldo Angelim, o 'Magro de Aço', que defendeu times como Fortaleza, Flamengo e Ituano. O zagueiro foi autor do gol do título do Flamengo no Campeonato Brasileiro de 2009. Categoria:Municípios do Ceará Categoria:Região do Cariri Categoria:Fundações no Ceará em 1953"
    text3 = bpe.decode(bpe.encode(text))
    print(text3 == text)
    print(bpe.tokenize(txt))
    print(len(bpe))
    try:
        if os.path.exists('/tmp/bpe'):
            os.remove('/tmp/bpe/vocabulary.dict')
            os.remove('/tmp/bpe/pairs.dict')
            os.remove('/tmp/bpe/merges.dict')
            os.rmdir('/tmp/bpe')
    except:
        pass
    bpe.save('/tmp/bpe')
    bpe2 = BPEBase()
    bpe2.load('/tmp/bpe')
    text3 = bpe2.decode(bpe2.encode(text))
    print(text3 == text)