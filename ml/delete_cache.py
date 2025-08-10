import paramiko
import sys
import os

def delete_remote_cache(host, user, private_key_path):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load private key
        try:
            key = paramiko.RSAKey.from_private_key_file(private_key_path)
        except paramiko.ssh_exception.SSHException:
            try:
                key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
            except paramiko.ssh_exception.SSHException:
                try:
                    key = paramiko.ECDSAKey.from_private_key_file(private_key_path)
                except Exception as e:
                    print(f"Error loading private key: {e}")
                    sys.exit(1)

        ssh.connect(hostname=host, username=user, pkey=key)

        # Command to delete cache
        command = "sudo rm -rf /var/cache/nginx/media/*"
        stdin, stdout, stderr = ssh.exec_command(command)
        
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print(f"Cache cleared on {host}")
        else:
            print(f"Error clearing cache on {host}: {stderr.read().decode()}")
            sys.exit(1)

    except Exception as e:
        print(f"SSH connection or command execution failed: {e}")
        sys.exit(1)
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python delete_cache.py <host> <user> <private_key_path>")
        sys.exit(1)

    host = sys.argv[1]
    user = sys.argv[2]
    private_key_path = sys.argv[3]

    delete_remote_cache(host, user, private_key_path)
