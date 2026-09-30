# Değerlendirme raporu — qwen2.5-7b (cpu)

- Embedding modeli: `qwen3-embedding-0.6b`
- top_k: 4, benzerlik eşiği: 0.53
- Parça sayısı: 240

## Arama (yalnızca cevaplanabilir sorular)

- Doğru doküman 1. sırada: 96%
- Doğru doküman ilk 4 içinde: 100%
- MRR: 0.981
- Cevap getirilen parçalarda mevcut: 100%

## Cevaplar

- Genel: 44/44 (yalnızca test bölümü: 22/22)
- Cevaplanabilir sorularda doğru cevap: 26/26 (yanlışlıkla reddedilen: 0)
- Verilen cevaplarda doğru dokümana atıf: 26/26
- Konu içi ama dokümanda olmayan sorular reddedildi: 10/10
- Konu dışı sorular reddedildi: 8/8
- Model çağrılmadan kapıda reddedilen: 10
- Model çağrılan sorularda ortanca süre: 13.61 sn (en uzun: 20.01 sn)

## Soru bazında sonuçlar

| # | Tür | Sonuç | Skor | Süre (sn) | Soru | Cevap |
|---|---|---|---|---|---|---|
| a01 | answerable | ✅ | 0.86 | 20.0 | Göbeklitepe hangi yıl UNESCO Dünya Mirası kalıcı listesine girdi? | Göbeklitepe 2018'de UNESCO Dünya Mirası kalıcı listesine girdi [K1]. |
| a02 | answerable | ✅ | 0.90 | 17.3 | Göbeklitepe hangi ilde bulunuyor? | Göbeklitepe, Türkiye'nin Güneydoğu Anadolu Bölgesi'ndeki Şanlıurfa ilinin 18 km kuzeydoğusunda yer alıyor [K1]. |
| a03 | answerable | ✅ | 0.84 | 16.5 | Göbeklitepe'de kazı çalışmalarını uzun yıllar kim yönetti? | Göbeklitepe'de kazı çalışmalarını uzun yıllar Klaus Schmidt yönetti [K4]. |
| a04 | answerable | ✅ | 0.65 | 15.3 | Efes'teki antik tiyatro kaç seyirci alabiliyordu? | Efes'teki antik tiyatro, II. Claudius döneminde başlayıp Trajan döneminde tamamlanarak 25.000 seyirci alabilen bir tiyatroydı [K1]. |
| a05 | answerable | ✅ | 0.79 | 14.1 | Efes hangi yıl Dünya Mirası olarak tescil edildi? | Efes 2015'te Dünya Mirası olarak tescil edildi [K1]. |
| a06 | answerable | ✅ | 0.87 | 12.2 | Hattuşa hangi uygarlığın başkentiydi? | Hattuşa Hititler'in başkentiydi [K1]. |
| a07 | answerable | ✅ | 0.69 | 8.5 | Hattuşa kaç yılında UNESCO Dünya Mirası listesine dahil edildi? | Hattuşaş, 1986 yılında UNESCO Dünya Mirasları listesine dâhil edilmiştir [K1]. |
| a08 | answerable | ✅ | 0.89 | 11.4 | Nemrut Dağı'nın yüksekliği kaç metredir? | Nemrut Dağı'nın yüksekliği 2.150 metre [K1]. |
| a09 | answerable | ✅ | 0.79 | 12.0 | Nemrut Dağı'nın tepesindeki mezar-tapınağı hangi kral yaptırdı? | Kommagene kralı Antiohos Theos, MÖ 62 yılında Nemrut Dağı'nın tepesine mezar-tapınağı yaptırdı [K1]. |
| a10 | answerable | ✅ | 0.76 | 16.9 | Troya antik kentini 1870'lerde kim keşfetti? | Troya antik kentini 1870'lerde Alman amatör arkeolog Heinrich Schliemann tarafından keşfetti [K4]. |
| a11 | answerable | ✅ | 0.67 | 18.8 | Troya hangi yıldan beri Dünya Mirası listesinde? | Troya, 1998 yılından beri Dünya Mirası listesinde yer almakta [K1]. |
| a12 | answerable | ✅ | 0.83 | 16.9 | Çatalhöyük'ü kim, hangi yıl keşfetti? | Çatalhöyük'ü James Mellaart, 1958 yılında keşfetti [K2]. |
| a13 | answerable | ✅ | 0.89 | 16.3 | Çatalhöyük Konya'nın hangi ilçesindedir? | Çatalhöyük, bugün Konya'nın Çumra ilçesine bağlı Küçükköy yakınlarında yer alır [K1]. |
| a14 | answerable | ✅ | 0.84 | 16.3 | Safranbolu hangi ile bağlıdır? | Safranbolu, Karabük ilinin tarihî ve turistik bir ilçesidir [K1]. |
| a15 | answerable | ✅ | 0.79 | 14.1 | Safranbolu hangi yıl Dünya Mirası listesine girdi? | Safranbolu 17 Aralık 1994 tarihinde Dünya Mirası listesine girmiştir [K1]. |
| a16 | answerable | ✅ | 0.80 | 17.7 | Divriği Ulu Camii'ni kim yaptırdı? | Divriği Ulu Cami 1228-29 yıllarında Mengücekli beyi Ahmed Şah tarafından yaptırılmıştır [K2]. |
| a17 | answerable | ✅ | 0.85 | 18.0 | Divriği Ulu Camii ve Darüşşifası hangi yıl Dünya Mirası listesine alındı? | Divriği Ulu Cami ve Darüşşifa, 1985 yılında Dünya Mirası listesine alınmıştır [K1]. |
| a18 | answerable | ✅ | 0.82 | 13.2 | Selimiye Camii'nin mimarı kimdir? | Selimiye Camii'nin mimarı Mimar Sinan'dır [K1]. |
| a19 | answerable | ✅ | 0.85 | 11.9 | Selimiye Camii hangi yıl ibadete açıldı? | Selimiye Camii 14 Mart 1575'te ibadete açılmıştır [K1]. |
| a20 | answerable | ✅ | 0.69 | 10.6 | Ani ören yeri hangi ilde bulunur? | Ani ören yeri Kars ilinde bulunur [K1]. |
| a21 | answerable | ✅ | 0.64 | 9.0 | Ani hangi yıl Dünya Mirası olarak tescil edildi? | Ani 2016'da Dünya Mirası olarak tescil edildi [K2]. |
| a22 | answerable | ✅ | 0.77 | 13.1 | Afrodisias hangi ilin hangi ilçesindedir? | Afrodisias, Anadolu'nun güneybatısında, eski Karia bölgesinde, günümüzde Aydın ilinin Karacasu ilçesine bağlı Geyre mahallesinde bulunan bir Antik Yunan kentidir [K1]. |
| a23 | answerable | ✅ | 0.73 | 13.9 | Afrodisias hangi yıl Dünya Mirası olarak tescil edildi? | Afrodisias 2017'de Dünya Mirası olarak tescil edildi [K1]. |
| a24 | answerable | ✅ | 0.84 | 10.1 | Pamukkale hangi ilde yer alır? | Pamukkale, Türkiye'nin güneybatısındaki Denizli ilinde yer alır [K1]. |
| a25 | answerable | ✅ | 0.79 | 10.0 | Pamukkale'de kaç adet sıcak su kaynağı bulunuyor? | Pamukkale'de 17 adet sıcak su kaynağı bulunuyor [K1]. |
| a26 | answerable | ✅ | 0.83 | 13.8 | Ani'deki Meryemana Kilisesi'ni hangi mimar inşa etti? | Ani'deki Meryemana Kilisesi, 989 yılında, İstanbul'daki Ayasofya'nın kubbesini onaran mimar Trdat tarafından inşa edilmiştir [K1]. |
| n01 | near_miss | ✅ | 0.62 | 11.1 | Efes'e giriş ücreti ne kadar? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n02 | near_miss | ✅ | 0.49 | 0.1 | Sümela Manastırı hangi ilde bulunur? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n03 | near_miss | ✅ | 0.54 | 14.2 | Aspendos Tiyatrosu kaç kişiliktir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n04 | near_miss | ✅ | 0.56 | 13.4 | Zeugma Mozaik Müzesi hangi şehirdedir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n05 | near_miss | ✅ | 0.51 | 0.2 | İshak Paşa Sarayı'nı kim yaptırmıştır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n06 | near_miss | ✅ | 0.80 | 12.6 | Göbeklitepe'ye en yakın havalimanı hangisidir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n07 | near_miss | ✅ | 0.65 | 9.8 | Pamukkale'ye giriş bileti kaç liradır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n08 | near_miss | ✅ | 0.56 | 8.7 | Bergama antik kenti hangi yıl Dünya Mirası listesine alındı? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n09 | near_miss | ✅ | 0.68 | 14.6 | Safranbolu lokumu nasıl yapılır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| n10 | near_miss | ✅ | 0.78 | 11.1 | Nemrut Dağı ören yerinin ziyaret saatleri nedir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o01 | out_of_domain | ✅ | 0.31 | 0.1 | Python'da bir liste nasıl sıralanır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o02 | out_of_domain | ✅ | 0.38 | 0.1 | En iyi pizza hamuru tarifi nedir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o03 | out_of_domain | ✅ | 0.41 | 0.1 | Fransa'nın başkenti neresidir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o04 | out_of_domain | ✅ | 0.43 | 0.1 | 2022 FIFA Dünya Kupası'nı hangi ülke kazandı? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o05 | out_of_domain | ✅ | 0.41 | 0.1 | Bir kilobayt kaç bayttır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o06 | out_of_domain | ✅ | 0.44 | 0.1 | Ay'a ilk ayak basan insan kimdir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o07 | out_of_domain | ✅ | 0.41 | 0.1 | Türk kahvesi nasıl pişirilir? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
| o08 | out_of_domain | ✅ | 0.35 | 0.1 | Mars'ın kaç uydusu vardır? | Bu bilgi elimdeki dokümanlarda yer almıyor. |
