import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PengirimPesan(Node):
    def __init__(self):
        # 1. Memberi nama pada Node 
        super().__init__('node_teriak')
        
        # 2. Membuat saluran pengumuman (Topik)
        self.publisher_ = self.create_publisher(String, 'saluran_gamaforce', 10)
        
        # 3. Membuat Timer setiap 1 detik
        jeda_waktu = 1.0  # dalam detik
        self.timer = self.create_timer(jeda_waktu, self.kirim_pesan)
        
        self.get_logger().info('Penyiar Radio Gamaforce sudah on-air!')

    def kirim_pesan(self):
        # 4. Membuat bungkus pesan dan mengisinya
        pesan = String()
        pesan.data = 'WOI DEK'
        
        # 5. Menyiarkan (publish) pesan ke topik
        self.publisher_.publish(pesan)
        
        # 6. Mencetak di terminal laptop
        self.get_logger().info('Saya baru saja berteriak: "%s"' % pesan.data)


def main(args=None):
    # Nyalakan mesin ROS 2
    rclpy.init(args=args)
    
    # Panggil kelas yang sudah kita buat di atas
    node_pengirim = PengirimPesan()
    
    # Biarkan program berputar terus-menerus (tidak langsung mati setelah 1 detik)
    rclpy.spin(node_pengirim)
    
    # Matikan mesin jika kita menekan Ctrl+C di terminal
    node_pengirim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()