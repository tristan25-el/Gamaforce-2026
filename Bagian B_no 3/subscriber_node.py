import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PenerimaPesan(Node):
    def __init__(self):
        # 1. Memberi nama pada Node 
        super().__init__('node_dengar')
        
        # 2. Membuat alat pendengar (Subscriber)
        self.subscription = self.create_subscription(
            String,
            'saluran_gamaforce',
            self.terima_pesan,
            10)
        self.subscription  # mencegah peringatan variabel tidak dipakai
        
        self.get_logger().info('Pendengar Radio Gamaforce sudah standby menunggu pesan!')

    def terima_pesan(self, pesan):
        # 3. Aksi saat pesan diterima: Cetak ke layar terminal
        self.get_logger().info('Saya mendengar: "%s"' % pesan.data)


def main(args=None):
    # Nyalakan mesin ROS 2
    rclpy.init(args=args)
    
    # Panggil kelas penerima
    node_penerima = PenerimaPesan()
    
    # Biarkan program terus-menerus menunggu pesan lewat
    rclpy.spin(node_penerima)
    
    # Matikan mesin jika ditekan Ctrl+C
    node_penerima.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()