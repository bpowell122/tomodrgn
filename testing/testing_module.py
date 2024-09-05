"""
Module to independently test multiple commands
"""

import os
import subprocess


class CommandTester:

    def __init__(self, workdir, verbose=False):
        self.commands = []
        self.stdout_file = f'{workdir}/commands.output'
        self.stderr_file = f'{workdir}/commands.error'
        self.successful_command_count = 0
        self.failed_command_count = 0
        self.failed_commands = []
        self.verbose = verbose

        print('------  SETUP  ------')
        print(f'All output files will be saved to: {os.path.abspath(workdir)}')
        print(f'Individual command output will be saved to {self.stdout_file}')
        print(f'Individual command errors will be saved to {self.stderr_file}')
        print('\n')

    def run(self):

        print('------  RUNNING  ------')
        for command_index, command in enumerate(self.commands):

            # convert to 1-based indexing for general legibility
            command_index += 1

            # run the command
            print(f'Running command {command_index}/{len(self.commands)}: ')
            print(f'    {command}')
            output = subprocess.run(command, capture_output=True, shell=True, text=True)
            if self.verbose:
                print(output.stdout)
                print(output.stderr)

            # write all stdout to logfile
            with open(self.stdout_file, 'a+') as f:
                f.write(f'Command {command_index}/{len(self.commands)}: {command} \n')
                f.write(f'{output.stdout} \n')

            # write all stderr to errfile if any exists
            if output.returncode == 0:
                print('    SUCCESS')
                self.successful_command_count += 1
            else:
                print(f'    ERROR: see {self.stderr_file}')
                self.failed_command_count += 1
                self.failed_commands.append(command)
                with open(self.stderr_file, 'a+') as f:
                    f.write(f'Command {command_index}/{len(self.commands)}: {command} \n')
                    f.write(f'Error returncode: {output.returncode} \n')
                    f.write(f'{output.stderr} \n')

        print('\n')

    def report_run_summary(self):
        print('------  SUMMARY  ------')
        print(f'{self.successful_command_count}/{len(self.commands)} completed successfully')
        print(f'{self.failed_command_count}/{len(self.commands)} failed with errors')
        if len(self.failed_commands) > 0:
            print('    Failed commands:')
            with open(self.stderr_file, 'r') as f:
                for line in f.readlines():
                    print(line.strip())
        print('\n')
