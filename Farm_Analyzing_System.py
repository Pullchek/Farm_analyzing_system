# Toprak görüntü analiz sistemi - Canlı Kamera Analizi
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

class ToprakAnalizSistemi:
    def __init__(self, model_path=None):
        """
        Toprak ve tarla analiz sistemi başlatma
        
        Parametreler:
        model_path : str, isteğe bağlı
            Eğitilmiş model dosyasının yolu
        """
        self.hastalik_model = None
        if model_path:
            self.hastalik_model = self.model_yukle(model_path)
        
    def goruntu_yukle(self, dosya_yolu):
        """
        Tarla görüntüsünü yükler
        
        Parametreler:
        dosya_yolu : str
            Görüntü dosyasının yolu
            
        Dönüş:
        np.array : Yüklenen görüntü
        """
        goruntu = cv2.imread(dosya_yolu)
        if goruntu is None:
            raise ValueError(f"{dosya_yolu} yüklenemedi")
            
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        return goruntu
    
    def nem_analizi(self, goruntu):
        """
        Görüntüden nem seviyesi tahmini yapar
        
        Parametreler:
        goruntu : np.array
            İşlenecek görüntü
            
        Dönüş:
        float : Tahmini nem oranı (%)
        dict : Detaylı nem bilgisi
        """
        # HSV dönüşümü nem analizi için daha uygundur
        hsv = cv2.cvtColor(goruntu, cv2.COLOR_RGB2HSV)
        
        # Mavi ve yeşil tonlarını analiz et (nem göstergesi)
        mavi_alt = np.array([90, 50, 50])
        mavi_ust = np.array([130, 255, 255])
        nem_maske = cv2.inRange(hsv, mavi_alt, mavi_ust)
        
        # Nem oranı hesaplama
        toplam_piksel = goruntu.shape[0] * goruntu.shape[1]
        nem_piksel = np.sum(nem_maske > 0)
        nem_orani = (nem_piksel / toplam_piksel) * 100
        
        # Nem seviyesi sınıflandırma
        nem_durumu = "Kuru"
        if nem_orani > 50:
            nem_durumu = "Çok Nemli"
        elif nem_orani > 30:
            nem_durumu = "Nemli"
        elif nem_orani > 15:
            nem_durumu = "Orta Nemli"
            
        return nem_orani, {
            "nem_orani": nem_orani,
            "nem_durumu": nem_durumu,
            "nem_haritasi": nem_maske
        }
    
    def hastalik_tespiti(self, goruntu):
        """
        Görüntüden hastalık tespiti yapar
        
        Parametreler:
        goruntu : np.array
            İşlenecek görüntü
            
        Dönüş:
        bool : Hastalık tespit edildi mi
        dict : Hastalık tespit detayları
        """
        # Renk özellikleri çıkarma
        hsv = cv2.cvtColor(goruntu, cv2.COLOR_RGB2HSV)
        
        # Hastalıklı bitki için yaygın renk aralıkları
        sari_alt = np.array([20, 100, 100])
        sari_ust = np.array([30, 255, 255])
        kahve_alt = np.array([10, 100, 20])
        kahve_ust = np.array([20, 255, 100])
        
        sari_maske = cv2.inRange(hsv, sari_alt, sari_ust)
        kahve_maske = cv2.inRange(hsv, kahve_alt, kahve_ust)
        
        # İki maskeyi birleştir
        hastalik_maske = cv2.bitwise_or(sari_maske, kahve_maske)
        
        # Hastalık oranı hesaplama
        toplam_piksel = goruntu.shape[0] * goruntu.shape[1]
        hastalik_piksel = np.sum(hastalik_maske > 0)
        hastalik_orani = (hastalik_piksel / toplam_piksel) * 100
        
        hastalik_var = hastalik_orani > 5.0  # %5'ten fazla hastalıklı alan varsa
        
        return hastalik_var, {
            "hastalik_orani": hastalik_orani,
            "hastalik_haritasi": hastalik_maske,
            "hastalik_turu": self._hastalik_siniflandirma(goruntu, hastalik_maske) if hastalik_var else "Yok"
        }
    
    def _hastalik_siniflandirma(self, goruntu, hastalik_maske):
        """
        Hastalık türünü tahmin etmeye çalışır (basit sürüm)
        
        Parametreler:
        goruntu : np.array
            Orijinal görüntü
        hastalik_maske : np.array
            Hastalıklı bölgelerin maskesi
            
        Dönüş:
        str : Tahmini hastalık türü
        """
        # Basit bir sınıflandırma (gerçek uygulamada daha karmaşık olmalı)
        # Bu örnekte sadece renk özellikleri kullanılıyor
        
        hastalikli_bolge = cv2.bitwise_and(goruntu, goruntu, mask=hastalik_maske)
        
        # Eğer hastalıklı maske boşsa (hiç hastalıklı piksel tespit edilmediyse)
        if np.sum(hastalik_maske) == 0:
            return "Yok"
        
        ortalama_renk = np.mean(hastalikli_bolge[hastalik_maske > 0], axis=0)
        
        # Örnek basit kural tabanlı sınıflandırma
        r, g, b = ortalama_renk
        if r > 150 and g > 100 and b < 80:
            return "Sarı Pas"
        elif r > 120 and g < 80 and b < 80:
            return "Kök Çürüklüğü"
        elif r < 100 and g > 100 and b < 100:
            return "Yaprak Küfü"
        else:
            return "Bilinmeyen Hastalık"
    
    def verim_analizi(self, goruntu):
        """
        Görüntüden verim tahmini yapar
        
        Parametreler:
        goruntu : np.array
            İşlenecek görüntü
            
        Dönüş:
        float : Tahmini verim skoru (0-100)
        dict : Verim analizi detayları
        """
        # Bitki örtüsü indeksi (NDVI benzeri bir yaklaşım)
        r = goruntu[:,:,0].astype(float)
        g = goruntu[:,:,1].astype(float)
        b = goruntu[:,:,2].astype(float)
        
        # Basitleştirilmiş bitki örtüsü indeksi
        # Gerçekte NDVI için NIR (yakın kızılötesi) bandı gereklidir
        bitki_indeksi = (g - r) / (g + r + 1e-10)  # Sıfıra bölmeyi önlemek için epsilon
        
        # -1 ile 1 arasındaki değerleri 0-100 aralığına ölçekle
        verim_haritasi = ((bitki_indeksi + 1) / 2 * 100).astype(np.uint8)
        
        # Ortalama verim skoru
        verim_skoru = np.mean(verim_haritasi)
        
        # Verim sınıflandırma
        verim_durumu = "Düşük"
        if verim_skoru > 70:
            verim_durumu = "Yüksek"
        elif verim_skoru > 50:
            verim_durumu = "Orta"
            
        return verim_skoru, {
            "verim_skoru": verim_skoru,
            "verim_durumu": verim_durumu,
            "verim_haritasi": verim_haritasi
        }
    
    def bolge_segmentasyonu(self, goruntu, bolge_sayisi=3):
        """
        Tarlayı farklı bölgelere ayırır
        
        Parametreler:
        goruntu : np.array
            İşlenecek görüntü
        bolge_sayisi : int
            Ayırmak istenen bölge sayısı
            
        Dönüş:
        np.array : Bölge etiketleri
        """
        # Görüntüyü düzleştir
        piksel_ozellikleri = goruntu.reshape(-1, 3)
        
        # K-means kümeleme
        kmeans = KMeans(n_clusters=bolge_sayisi, random_state=42, n_init=10)
        etiketler = kmeans.fit_predict(piksel_ozellikleri)
        
        # Etiketleri orijinal görüntü şekline dönüştür
        segmente_goruntu = etiketler.reshape(goruntu.shape[0], goruntu.shape[1])
        
        return segmente_goruntu
    
    def goruntu_analiz(self, goruntu):
        """
        Verilen görüntüyü analiz eder ve sonuçları döndürür
        
        Parametreler:
        goruntu : np.array
            İşlenecek görüntü
            
        Dönüş:
        dict : Analiz sonuçları
        """
        # BGR'dan RGB'ye dönüştür (OpenCV BGR formatında alır)
        if len(goruntu.shape) == 3 and goruntu.shape[2] == 3:
            goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        else:
            goruntu_rgb = goruntu
            
        # Tüm analizleri gerçekleştir
        nem_orani, nem_detay = self.nem_analizi(goruntu_rgb)
        hastalik_var, hastalik_detay = self.hastalik_tespiti(goruntu_rgb)
        verim_skoru, verim_detay = self.verim_analizi(goruntu_rgb)
        
        return {
            "nem_durumu": nem_detay['nem_durumu'],
            "nem_orani": nem_orani,
            "nem_haritasi": nem_detay['nem_haritasi'],
            "hastalik_var": hastalik_var,
            "hastalik_orani": hastalik_detay['hastalik_orani'],
            "hastalik_turu": hastalik_detay['hastalik_turu'],
            "hastalik_haritasi": hastalik_detay['hastalik_haritasi'],
            "verim_skoru": verim_skoru,
            "verim_durumu": verim_detay['verim_durumu'],
            "verim_haritasi": verim_detay['verim_haritasi']
        }

    def canli_kamera_analizi(self, kamera_id=0, analiz_siklik=1):
        """
        Webcam veya kameradan canlı görüntü alarak analiz yapar
        
        Parametreler:
        kamera_id : int, default 0
            Kullanılacak kamera ID'si (birden fazla kamera varsa)
        analiz_siklik : int, default 1
            Kaç saniyede bir analiz yapılacağı
            
        Dönüş:
        None
        """
        # Kamerayı başlat
        kamera = cv2.VideoCapture(kamera_id)
        
        if not kamera.isOpened():
            print("Kamera açılamadı!")
            return
        
        print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")
        
        son_analiz_zamani = time.time() - analiz_siklik  # İlk kare için hemen analiz yap
        sonuclar = None
        
        while True:
            # Kameradan kare al
            ret, kare = kamera.read()
            
            if not ret:
                print("Kare alınamadı!")
                break
                
            # Görüntüyü ekranda göster (orijinal görüntü)
            cv2.imshow("Toprak Analiz Sistemi - Orijinal", kare)
            
            # Analiz zamanı geldiyse analiz yap
            simdiki_zaman = time.time()
            if simdiki_zaman - son_analiz_zamani >= analiz_siklik:
                son_analiz_zamani = simdiki_zaman
                
                # Görüntüyü analiz et
                sonuclar = self.goruntu_analiz(kare)
                
                # Görüntü üzerine bilgileri yaz
                analiz_kare = kare.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Nem bilgisi
                cv2.putText(analiz_kare, f"Nem: {sonuclar['nem_durumu']} (%{sonuclar['nem_orani']:.1f})", 
                            (10, 30), font, 0.7, (255, 255, 255), 2)
                
                # Hastalık bilgisi
                if sonuclar['hastalik_var']:
                    cv2.putText(analiz_kare, f"Hastalık: {sonuclar['hastalik_turu']} (%{sonuclar['hastalik_orani']:.1f})", 
                                (10, 60), font, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(analiz_kare, "Hastalık: YOK", 
                                (10, 60), font, 0.7, (0, 255, 0), 2)
                
                # Verim bilgisi
                cv2.putText(analiz_kare, f"Verim: {sonuclar['verim_durumu']} ({sonuclar['verim_skoru']:.1f}/100)", 
                            (10, 90), font, 0.7, (255, 255, 255), 2)
                
                # Analiz sonuçlarını göster
                cv2.imshow("Toprak Analiz Sistemi - Sonuçlar", analiz_kare)
                
                # Nem haritasını göster
                cv2.imshow("Nem Haritası", sonuclar['nem_haritasi'])
                
                # Hastalık haritasını göster
                cv2.imshow("Hastalık Haritası", sonuclar['hastalik_haritasi'])
                
                # Verim haritasını göster
                verim_renkli = cv2.applyColorMap(sonuclar['verim_haritasi'], cv2.COLORMAP_JET)
                cv2.imshow("Verim Haritası", verim_renkli)
                
            # Klavyeden tuş kontrolü
            tus = cv2.waitKey(1) & 0xFF
            if tus == ord('q'):
                break
            elif tus == ord('s'):  # Anlık görüntü kaydet
                zaman_damgasi = time.strftime("%Y%m%d-%H%M%S")
                dosya_adi = f"toprak_analiz_{zaman_damgasi}.jpg"
                cv2.imwrite(dosya_adi, kare)
                print(f"Görüntü kaydedildi: {dosya_adi}")
                
                # Kayıtlı görüntü üzerinde tam rapor oluştur
                if sonuclar is not None:
                    goruntu_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
                    rapor_adi = f"rapor_{zaman_damgasi}.png"
                    self.rapor_olustur_canli(goruntu_rgb, sonuclar, rapor_adi)
                    print(f"Rapor oluşturuldu: {rapor_adi}")
        
        # Temizlik
        kamera.release()
        cv2.destroyAllWindows()
    
    def rapor_olustur_canli(self, goruntu, sonuclar, dosya_adi="toprak_analiz_raporu.png"):
        """
        Canlı kamera analizi için tam rapor oluşturur ve kaydeder
        
        Parametreler:
        goruntu : np.array
            Orijinal görüntü
        sonuclar : dict
            Analiz sonuçları
        dosya_adi : str
            Kaydedilecek rapor dosyasının adı
        """
        # Bölge segmentasyonu sadece rapor için yapılır
        bolge_haritasi = self.bolge_segmentasyonu(goruntu)
        
        # Sonuçları görselleştir
        plt.figure(figsize=(14, 10))
        
        # Orijinal görüntü
        plt.subplot(2, 3, 1)
        plt.imshow(goruntu)
        plt.title("Orijinal Görüntü")
        plt.axis('off')
        
        # Nem haritası
        plt.subplot(2, 3, 2)
        plt.imshow(sonuclar['nem_haritasi'], cmap='Blues')
        plt.title(f"Nem Haritası: %{sonuclar['nem_orani']:.1f}")
        plt.axis('off')
        
        # Hastalık haritası
        plt.subplot(2, 3, 3)
        plt.imshow(sonuclar['hastalik_haritasi'], cmap='Reds')
        plt.title(f"Hastalık Haritası: %{sonuclar['hastalik_orani']:.1f}")
        plt.axis('off')
        
        # Verim haritası
        plt.subplot(2, 3, 4)
        plt.imshow(sonuclar['verim_haritasi'], cmap='Greens')
        plt.title(f"Verim Haritası: {sonuclar['verim_skoru']:.1f}/100")
        plt.axis('off')
        
        # Bölge segmentasyonu
        plt.subplot(2, 3, 5)
        plt.imshow(bolge_haritasi, cmap='tab10')
        plt.title("Toprak Bölge Analizi")
        plt.axis('off')
        
        # Özet bilgi metni
        plt.subplot(2, 3, 6)
        plt.axis('off')
        ozet_metin = f"""
        TOPRAK VE TARLA ANALİZ RAPORU
        
        Nem Durumu: {sonuclar['nem_durumu']} (%{sonuclar['nem_orani']:.1f})
        
        Hastalık Durumu: {"VAR" if sonuclar['hastalik_var'] else "YOK"}
        Hastalık Oranı: %{sonuclar['hastalik_orani']:.1f}
        Hastalık Türü: {sonuclar['hastalik_turu']}
        
        Verim Skoru: {sonuclar['verim_skoru']:.1f}/100
        Verim Durumu: {sonuclar['verim_durumu']}
        
        Öneriler:
        {"- Hastalık tedavisi gerekli!" if sonuclar['hastalik_var'] else "- Hastalık bulunmadı."}
        {"- Sulama azaltılmalı!" if sonuclar['nem_orani'] > 50 else "- Sulama artırılmalı!" if sonuclar['nem_orani'] < 15 else "- Sulama uygun."}
        {"- Gübreleme artırılmalı!" if sonuclar['verim_skoru'] < 40 else "- Gübreleme uygun."}
        """
        plt.text(0, 0.5, ozet_metin, fontsize=10, va='center')
        
        # Raporu kaydet
        plt.tight_layout()
        plt.savefig(dosya_adi)
        plt.close()
        
        return dosya_adi

# Kullanım örneği
if __name__ == "__main__":
    analiz_sistemi = ToprakAnalizSistemi()
    
    # Canlı kamera analizi başlat
    print("Toprak Analiz Sistemi - Canlı Kamera Modu")
    print("-" * 40)
    print("Kullanım:")
    print("- 'q' tuşu: Programdan çık")
    print("- 's' tuşu: Anlık görüntü ve rapor kaydet")
    print("-" * 40)
    
    # Kamera başlat (0: varsayılan kamera, değiştirilebilir)
    # analiz_siklik parametresi kaç saniyede bir analiz yapılacağını belirler (varsayılan 1 saniye)
    analiz_sistemi.canli_kamera_analizi(kamera_id=0, analiz_siklik=1)